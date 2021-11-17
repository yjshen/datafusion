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

use crate::error::DataFusionError;
use crate::physical_plan::expressions::{
    exprs_to_sort_columns, Column, PhysicalSortExpr,
};
use crate::physical_plan::joins::equal_rows;
use crate::physical_plan::SendableRecordBatchStream;
use arrow::array::ArrayRef;
use arrow::compute::partition::lexicographical_partition_ranges;
use arrow::error::ArrowError;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use std::collections::VecDeque;
use std::ops::Range;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(crate) fn join_arrays(rb: &RecordBatch, on_column: &Vec<Column>) -> Vec<ArrayRef> {
    on_column
        .iter()
        .map(|c| rb.column(c.index()).clone())
        .collect()
}

#[derive(Clone)]
struct PartitionedRecordBatch {
    batch: RecordBatch,
    ranges: Vec<Range<usize>>,
}

impl PartitionedRecordBatch {
    fn new(
        batch: Option<RecordBatch>,
        expr: &[PhysicalSortExpr],
    ) -> ArrowResult<Option<Self>> {
        match batch {
            Some(batch) => {
                let columns = exprs_to_sort_columns(&batch, expr)
                    .map_err(DataFusionError::into_arrow_external_error)?;
                let ranges = lexicographical_partition_ranges(
                    &columns.iter().map(|x| x.into()).collect::<Vec<_>>(),
                )?
                .collect::<Vec<_>>();
                Ok(Some(Self { batch, ranges }))
            }
            None => Ok(None),
        }
    }

    #[inline]
    pub fn is_last_range(&self, range: &Range<usize>) -> bool {
        range.end == self.batch.num_rows()
    }
}

struct StreamingSideBuffer {
    batch: Option<PartitionedRecordBatch>,
    cur_row: usize,
    cur_range: usize,
    num_rows: usize,
    num_ranges: usize,
    is_new_key: bool,
    on_column: Vec<Column>,
    sort: Vec<PhysicalSortExpr>,
}

impl StreamingSideBuffer {
    fn new(on_column: Vec<Column>, sort: Vec<PhysicalSortExpr>) -> Self {
        Self {
            batch: None,
            cur_row: 0,
            cur_range: 0,
            num_rows: 0,
            num_ranges: 0,
            is_new_key: true,
            on_column,
            sort,
        }
    }

    fn reset(&mut self, prb: Option<PartitionedRecordBatch>) {
        self.batch = prb;
        if let Some(prb) = &self.batch {
            self.cur_row = 0;
            self.cur_range = 0;
            self.num_rows = prb.batch.num_rows();
            self.num_ranges = prb.ranges.len();
            self.is_new_key = true;
        };
    }

    fn key_any_null(&self) -> bool {
        match &self.batch {
            None => return true,
            Some(batch) => {
                for c in self.on_column {
                    let array = batch.batch.column(c.index());
                    if array.is_null(self.cur_row) {
                        return true;
                    }
                }
                false
            }
        }
    }

    #[inline]
    fn is_finished(&self) -> bool {
        self.batch.is_none() || self.num_rows == self.cur_row + 1
    }

    #[inline]
    fn is_last_key_in_batch(&self) -> bool {
        self.batch.is_none() || self.num_ranges == self.cur_range + 1
    }

    fn advance(&mut self) {
        self.cur_row += 1;
        self.is_new_key = false;
        if !self.is_last_key_in_batch() {
            let ranges = self.batch.unwrap().ranges;
            if self.cur_row == ranges[self.cur_range + 1].start {
                self.cur_range += 1;
                self.is_new_key = true;
            }
        }
    }

    fn advance_key(&mut self) {
        let ranges = self.batch.unwrap().ranges;
        self.cur_range += 1;
        self.cur_row = ranges[self.cur_range].start;
        self.is_new_key = true;
    }
}

struct StreamingSideStream {
    input: SendableRecordBatchStream,
    output: StreamingSideBuffer,
    input_is_finished: bool,
    sort: Vec<PhysicalSortExpr>,
}

impl StreamingSideStream {
    fn inner_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.input_is_finished {
            Poll::Ready(None)
        } else {
            match self.input.poll_next_unpin(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(x) => match x {
                    None => {
                        self.input_is_finished = true;
                        Poll::Ready(None)
                    }
                    batch => {
                        let batch = batch.transpose()?;
                        let prb = PartitionedRecordBatch::new(batch, &self.sort)?;
                        self.output.reset(prb);
                        Poll::Ready(Some(Ok(())))
                    }
                },
            }
        }
    }

    fn advance(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.output.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            if self.output.is_finished() {
                match self.inner_next(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(x) => match x {
                        None => Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if self.output.is_finished() {
                                Poll::Ready(None)
                            } else {
                                Poll::Ready(Some(Ok(())))
                            }
                        }
                    },
                }
            } else {
                self.output.advance();
                Poll::Ready(Some(Ok(())))
            }
        }
    }

    fn advance_key(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.output.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            if self.output.is_finished() || self.output.is_last_key_in_batch() {
                match self.inner_next(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(x) => match x {
                        None => Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if self.output.is_finished() {
                                Poll::Ready(None)
                            } else {
                                Poll::Ready(Some(Ok(())))
                            }
                        }
                    },
                }
            } else {
                self.output.advance_key();
                Poll::Ready(Some(Ok(())))
            }
        }
    }

    fn advance_key_skip_null(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.output.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            loop {
                match self.advance_key(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if !self.output.key_any_null() {
                                return Poll::Ready(Some(Ok(())));
                            }
                        }
                    },
                    Poll::Pending => return Poll::Pending,
                }
            }
        }
    }
}

/// Holding ranges for same key over several bathes
struct BufferingSideBuffer {
    /// batches that contains the current key
    /// TODO: make this spillable as well for skew on join key at buffer side
    batches: VecDeque<PartitionedRecordBatch>,
    /// ranges in each PartitionedRecordBatch that contains the current key
    ranges: VecDeque<Range<usize>>,
    /// row index in first batch to the record that starts this batch
    key_idx: Option<usize>,
    /// total number of rows for the current key
    row_num: usize,
    /// hold found but not currently used batch, to continue iteration
    next_key_batch: Option<PartitionedRecordBatch>,
    /// Join on column
    on_column: Vec<Column>,
    sort: Vec<PhysicalSortExpr>,
}

#[inline]
pub(crate) fn range_len(range: &Range<usize>) -> usize {
    range.end - range.start
}

impl BufferingSideBuffer {
    fn new(on_column: Vec<Column>, sort: Vec<PhysicalSortExpr>) -> Self {
        Self {
            batches: VecDeque::new(),
            ranges: VecDeque::new(),
            key_idx: None,
            row_num: 0,
            next_key_batch: None,
            on_column,
            sort,
        }
    }

    fn key_any_null(&self) -> bool {
        match &self.key_idx {
            None => return true,
            Some(key_idx) => {
                let first_batch = &self.batches[0].batch;
                for c in self.on_column {
                    let array = first_batch.column(c.index());
                    if array.is_null(*key_idx) {
                        return true;
                    }
                }
                false
            }
        }
    }

    fn is_finished(&self) -> ArrowResult<bool> {
        match self.key_idx {
            None => Ok(true),
            Some(_) => match (self.batches.back(), self.ranges.back()) {
                (Some(batch), Some(range)) => Ok(batch.is_last_range(range)),
                _ => Err(ArrowError::Other(format!(
                    "Batches length {} not equal to ranges length {}",
                    self.batches.len(),
                    self.ranges.len()
                ))),
            },
        }
    }

    /// Whether the running key ends at the current batch `prb`, true for continues, false for ends.
    fn running_key(&mut self, prb: &PartitionedRecordBatch) -> ArrowResult<bool> {
        let first_range = &prb.ranges[0];
        let range_len = range_len(first_range);
        let current_batch = &prb.batch;
        let single_range = prb.ranges.len() == 1;

        // compare the first record in batch with the current key pointed by key_idx
        match self.key_idx {
            None => {
                self.batches.push_back(prb.clone());
                self.ranges.push_back(first_range.clone());
                self.key_idx = Some(0);
                self.row_num += range_len;
                Ok(single_range)
            }
            Some(key_idx) => {
                let key_arrays = join_arrays(&self.batches[0].batch, &self.on_column);
                let current_arrays = join_arrays(current_batch, &self.on_column);
                let equal = equal_rows(key_idx, 0, &key_arrays, &current_arrays)
                    .map_err(DataFusionError::into_arrow_external_error)?;
                if equal {
                    self.batches.push_back(prb.clone());
                    self.ranges.push_back(first_range.clone());
                    self.row_num += range_len;
                    Ok(single_range)
                } else {
                    self.next_key_batch = Some(prb.clone());
                    Ok(false) // running key ends
                }
            }
        }
    }

    fn cleanup(&mut self) {
        self.batches.drain(..);
        self.ranges.drain(..);
        self.next_key_batch = None;
    }

    fn reset(&mut self, prb: &PartitionedRecordBatch) {
        self.cleanup();
        self.batches.push_back(prb.clone());
        let first_range = &prb.ranges[0];
        self.ranges.push_back(first_range.clone());
        self.key_idx = Some(0);
        self.row_num = range_len(first_range);
    }

    /// Advance the cursor to the next key seen by this buffer
    fn advance_key(&mut self) {
        assert_eq!(self.batches.len(), self.ranges.len());
        if self.batches.len() > 1 {
            self.batches.drain(0..(self.batches.len() - 1));
            self.ranges.drain(0..(self.batches.len() - 1));
        }

        if let Some(batch) = self.batches.pop_back() {
            let tail_range = self.ranges.pop_back().unwrap();
            self.batches.push_back(batch);
            let next_range_idx = batch
                .ranges
                .iter()
                .enumerate()
                .find(|(idx, range)| range.start == tail_range.start)
                .unwrap()
                .0;
            self.key_idx = Some(tail_range.end);
            self.ranges.push_back(batch.ranges[next_range_idx].clone());
            self.row_num = range_len(&batch.ranges[next_range_idx]);
        }
    }
}

struct BufferingSideStream {
    input: SendableRecordBatchStream,
    output: BufferingSideBuffer,
    input_is_finished: bool,
    cumulating: bool,
    sort: Vec<PhysicalSortExpr>,
}

impl BufferingSideStream {
    fn inner_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<PartitionedRecordBatch>>> {
        if self.input_is_finished {
            Poll::Ready(None)
        } else {
            match self.input.poll_next_unpin(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(x) => match x {
                    None => {
                        self.input_is_finished = true;
                        Poll::Ready(None)
                    }
                    batch => {
                        let batch = batch.transpose()?;
                        let prb =
                            PartitionedRecordBatch::new(batch, &self.sort).transpose();
                        Poll::Ready(prb)
                    }
                },
            }
        }
    }

    fn advance_key(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.output.is_finished()? && self.input_is_finished {
            return Poll::Ready(None);
        } else {
            if self.cumulating {
                match self.cumulate_same_keys(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(x) => match x {
                        Some(x) => {
                            x?;
                            return Poll::Ready(Some(Ok(())));
                        }
                        None => unreachable!(),
                    },
                }
            }

            if self.output.is_finished()? {
                return match &self.output.next_key_batch {
                    None => match self.inner_next(cx) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(x) => match x {
                            None => Poll::Ready(None),
                            Some(x) => {
                                let prb = x?;
                                self.output.reset(&prb);
                                if prb.ranges.len() == 1 {
                                    self.cumulating = true;
                                    Poll::Pending
                                } else {
                                    Poll::Ready(Some(Ok(())))
                                }
                            }
                        },
                    },
                    Some(batch) => {
                        self.output.reset(batch);
                        if batch.ranges.len() == 1 {
                            self.cumulating = true;
                            Poll::Pending
                        } else {
                            Poll::Ready(Some(Ok(())))
                        }
                    }
                };
            } else {
                self.output.advance_key();
                if self.output.batches[0].is_last_range(&self.output.ranges[0]) {
                    self.cumulating = true;
                } else {
                    return Poll::Ready(Some(Ok(())));
                }
            }
        }

        unreachable!()
    }

    fn advance_key_skip_null(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.output.is_finished()? && self.input_is_finished {
            Poll::Ready(None)
        } else {
            loop {
                match self.advance_key(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if !self.output.key_any_null() {
                                return Poll::Ready(Some(Ok(())));
                            }
                        }
                    },
                    Poll::Pending => return Poll::Pending,
                }
            }
        }
    }

    fn cumulate_same_keys(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        loop {
            match self.inner_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(x) => match x {
                    None => {
                        self.cumulating = false;
                        return Poll::Ready(Some(Ok(())));
                    }
                    Some(x) => {
                        let prb = x?;
                        let buffer_more = self.output.running_key(&prb)?;
                        if !buffer_more {
                            self.cumulating = false;
                            return Poll::Ready(Some(Ok(())));
                        }
                    }
                },
            }
        }
    }
}
