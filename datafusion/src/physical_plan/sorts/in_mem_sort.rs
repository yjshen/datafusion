// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use super::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use std::any::Any;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::ord::DynComparator;
use arrow::array::{growable::make_growable, ord::build_compare, ArrayRef};
use arrow::compute::sort::SortOptions;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::stream::FusedStream;
use futures::{Stream, StreamExt};
use hashbrown::HashMap;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::sorts::{RowIndex, SortKeyCursor, SortKeyCursorWrapper};
use crate::physical_plan::{
    common::spawn_execution, expressions::PhysicalSortExpr, DisplayFormatType,
    Distribution, ExecutionPlan, Partitioning, PhysicalExpr, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};

#[derive(Debug)]
pub(crate) struct InMemSortStream<'a, 'b> {
    /// The schema of the RecordBatches yielded by this stream
    schema: SchemaRef,
    /// For each input stream maintain a dequeue of SortKeyCursor
    ///
    /// Exhausted cursors will be popped off the front once all
    /// their rows have been yielded to the output
    cursors: Vec<SortKeyCursor>,
    /// The accumulated row indexes for the next record batch
    in_progress: Vec<RowIndex>,
    /// The physical expressions to sort by
    column_expressions: Vec<Arc<dyn PhysicalExpr>>,
    /// The sort options for each expression
    sort_options: Vec<SortOptions>,
    /// The desired RecordBatch size to yield
    target_batch_size: usize,
    /// used to record execution metrics
    baseline_metrics: BaselineMetrics,
    /// If the stream has encountered an error
    aborted: bool,
    /// min heap for record comparison
    min_heap: BinaryHeap<SortKeyCursorWrapper<'a, 'b>>,
}

impl<'a, 'b> InMemSortStream {
    pub(crate) fn new(
        mut sorted_batches: Vec<RecordBatch>,
        schema: SchemaRef,
        expressions: &[PhysicalSortExpr],
        target_batch_size: usize,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        let len = sorted_batches.len();
        let mut cursors = Vec::with_capacity(len);
        let mut min_heap = BinaryHeap::with_capacity(len);

        let column_expressions: Vec<Arc<dyn PhysicalExpr>> =
            expressions.iter().map(|x| x.expr.clone()).collect();

        let sort_options: Vec<SortOptions> =
            expressions.iter().map(|x| x.options).collect();

        sorted_batches
            .into_iter()
            .enumerate()
            .try_for_each(|(idx, batch)| {
                let cursor = match SortKeyCursor::new(idx, batch, &column_expressions) {
                    Ok(cursor) => cursor,
                    Err(e) => return Err(e),
                };
                let wrapper = SortKeyCursorWrapper::new(&cursor, &sort_options);
                min_heap.push(wrapper);
                cursors[idx] = cursor;
                Ok(())
            })?;

        Ok(Self {
            schema,
            cursors,
            column_expressions,
            sort_options,
            target_batch_size,
            baseline_metrics,
            aborted: false,
            in_progress: vec![],
            min_heap,
        })
    }

    /// Returns the index of the next batch to pull a row from, or None
    /// if all cursors for all batch are exhausted
    fn next_batch_idx(&mut self) -> Result<Option<usize>> {
        match self.min_heap.pop() {
            None => Ok(None),
            Some(batch) => Ok(Some(batch.cursor.batch_idx)),
        }
    }

    /// Drains the in_progress row indexes, and builds a new RecordBatch from them
    ///
    /// Will then drop any cursors for which all rows have been yielded to the output
    fn build_record_batch(&mut self) -> ArrowResult<RecordBatch> {
        let columns = self
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(column_idx, _)| {
                let arrays = self
                    .cursors
                    .iter()
                    .map(|cursor| cursor.batch.column(column_idx).as_ref())
                    .collect::<Vec<_>>();

                let mut array_data =
                    make_growable(&arrays, false, self.in_progress.len());

                if self.in_progress.is_empty() {
                    return array_data.as_arc();
                }

                let first = &self.in_progress[0];
                let mut buffer_idx = first.stream_idx;
                let mut start_row_idx = first.row_idx;
                let mut end_row_idx = start_row_idx + 1;

                for row_index in self.in_progress.iter().skip(1) {
                    let next_buffer_idx = row_index.stream_idx;

                    if next_buffer_idx == buffer_idx && row_index.row_idx == end_row_idx {
                        // subsequent row in same batch
                        end_row_idx += 1;
                        continue;
                    }

                    // emit current batch of rows for current buffer
                    array_data.extend(
                        buffer_idx,
                        start_row_idx,
                        end_row_idx - start_row_idx,
                    );

                    // start new batch of rows
                    buffer_idx = next_buffer_idx;
                    start_row_idx = row_index.row_idx;
                    end_row_idx = start_row_idx + 1;
                }

                // emit final batch of rows
                array_data.extend(buffer_idx, start_row_idx, end_row_idx - start_row_idx);
                array_data.as_arc()
            })
            .collect();

        self.in_progress.clear();
        RecordBatch::try_new(self.schema.clone(), columns)
    }

    #[inline]
    fn poll_next_inner(
        self: &mut Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<RecordBatch>>> {
        if self.aborted {
            return Poll::Ready(None);
        }

        loop {
            // NB timer records time taken on drop, so there are no
            // calls to `timer.done()` below.
            let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
            let _timer = elapsed_compute.timer();

            let batch_idx = match self.next_batch_idx() {
                Ok(Some(idx)) => idx,
                Ok(None) if self.in_progress.is_empty() => return Poll::Ready(None),
                Ok(None) => return Poll::Ready(Some(self.build_record_batch())),
                Err(e) => {
                    self.aborted = true;
                    return Poll::Ready(Some(Err(ArrowError::External(
                        "".to_string(),
                        Box::new(e),
                    ))));
                }
            };

            let cursor = &mut self.cursors[batch_idx];
            let row_idx = cursor.advance();

            // insert the cursor back to min_heap if the record batch is not exhausted
            if !cursor.is_finished() {
                self.min_heap
                    .push(SortKeyCursorWrapper::new(cursor, &self.sort_options));
            }

            self.in_progress.push(RowIndex {
                stream_idx: batch_idx,
                cursor_idx: 0,
                row_idx,
            });

            if self.in_progress.len() == self.target_batch_size {
                return Poll::Ready(Some(self.build_record_batch()));
            }
        }
    }
}

impl<'a, 'b> Stream for InMemSortStream<'a, 'b> {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.poll_next_inner(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

impl<'a, 'b> RecordBatchStream for InMemSortStream<'a, 'b> {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
