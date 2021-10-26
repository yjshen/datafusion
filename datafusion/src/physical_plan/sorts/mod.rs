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

//! Sort functionalities

pub mod external_sort;
mod in_mem_sort;
pub mod sort;
pub mod sort_preserving_merge;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{PhysicalExpr, RecordBatchStream, SendableRecordBatchStream};
use arrow::array::ord::{build_compare, DynComparator};
pub use arrow::compute::sort::SortOptions;
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, error::ArrowError, error::Result as ArrowResult};
use futures::channel::mpsc;
use futures::stream::FusedStream;
use futures::Stream;
use hashbrown::HashMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// A `SortKeyCursor` is created from a `RecordBatch`, and a set of
/// `PhysicalExpr` that when evaluated on the `RecordBatch` yield the sort keys.
///
/// Additionally it maintains a row cursor that can be advanced through the rows
/// of the provided `RecordBatch`
///
/// `SortKeyCursor::compare` can then be used to compare the sort key pointed to
/// by this row cursor, with that of another `SortKeyCursor`. A cursor stores
/// a row comparator for each other cursor that it is compared to.
struct SortKeyCursor {
    columns: Vec<ArrayRef>,
    cur_row: usize,
    num_rows: usize,

    // An index uniquely identifying the record batch scanned by this cursor.
    batch_idx: usize,
    batch: RecordBatch,

    // A collection of comparators that compare rows in this cursor's batch to
    // the cursors in other batches. Other batches are uniquely identified by
    // their batch_idx.
    batch_comparators: HashMap<usize, Vec<DynComparator>>,
}

impl std::fmt::Debug for SortKeyCursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortKeyCursor")
            .field("columns", &self.columns)
            .field("cur_row", &self.cur_row)
            .field("num_rows", &self.num_rows)
            .field("batch_idx", &self.batch_idx)
            .field("batch", &self.batch)
            .field("batch_comparators", &"<FUNC>")
            .finish()
    }
}

impl SortKeyCursor {
    fn new(
        batch_idx: usize,
        batch: RecordBatch,
        sort_key: &[Arc<dyn PhysicalExpr>],
    ) -> Result<Self> {
        let columns: Vec<ArrayRef> = sort_key
            .iter()
            .map(|expr| Ok(expr.evaluate(&batch)?.into_array(batch.num_rows())))
            .collect::<Result<_>>()?;
        assert_eq!(columns.len(), sort_options.len(),
                   "Incorrect number of SortOptions provided to SortKeyCursor::compare, expected {} got {}",
                   columns.len(), sort_options.len());
        Ok(Self {
            cur_row: 0,
            num_rows: batch.num_rows(),
            columns,
            batch,
            batch_idx,
            batch_comparators: HashMap::new(),
        })
    }

    fn is_finished(&self) -> bool {
        self.num_rows == self.cur_row
    }

    fn advance(&mut self) -> usize {
        assert!(!self.is_finished());
        let t = self.cur_row;
        self.cur_row += 1;
        t
    }

    /// Compares the sort key pointed to by this instance's row cursor with that of another
    fn compare(
        &mut self,
        other: &SortKeyCursor,
        options: &[SortOptions],
    ) -> Result<Ordering> {
        if self.columns.len() != other.columns.len() {
            return Err(DataFusionError::Internal(format!(
                "SortKeyCursors had inconsistent column counts: {} vs {}",
                self.columns.len(),
                other.columns.len()
            )));
        }

        if self.columns.len() != options.len() {
            return Err(DataFusionError::Internal(format!(
                "Incorrect number of SortOptions provided to SortKeyCursor::compare, expected {} got {}",
                self.columns.len(),
                options.len()
            )));
        }

        let zipped = self
            .columns
            .iter()
            .zip(other.columns.iter())
            .zip(options.iter());

        // Recall or initialise a collection of comparators for comparing
        // columnar arrays of this cursor and "other".
        let cmp = self
            .batch_comparators
            .entry(other.batch_idx)
            .or_insert_with(|| Vec::with_capacity(other.columns.len()));

        for (i, ((l, r), sort_options)) in zipped.enumerate() {
            if i >= cmp.len() {
                // initialise comparators as potentially needed
                cmp.push(arrow::array::build_compare(l.as_ref(), r.as_ref())?);
            }

            match (l.is_valid(self.cur_row), r.is_valid(other.cur_row)) {
                (false, true) if sort_options.nulls_first => return Ok(Ordering::Less),
                (false, true) => return Ok(Ordering::Greater),
                (true, false) if sort_options.nulls_first => {
                    return Ok(Ordering::Greater)
                }
                (true, false) => return Ok(Ordering::Less),
                (false, false) => {}
                (true, true) => match cmp[i](self.cur_row, other.cur_row) {
                    Ordering::Equal => {}
                    o if sort_options.descending => return Ok(o.reverse()),
                    o => return Ok(o),
                },
            }
        }

        Ok(Ordering::Equal)
    }
}

/// A `RowIndex` identifies a specific row from those buffered
/// by a `SortPreservingMergeStream`
#[derive(Debug, Clone)]
struct RowIndex {
    /// The index of the stream
    stream_idx: usize,
    /// For sort_preserving_merge, it's the index of the cursor within the stream's VecDequeue.
    /// For in_mem_sort which have only one batch for each stream, cursor_idx always 0
    cursor_idx: usize,
    /// The row index
    row_idx: usize,
}

pub struct SortKeyCursorWrapper<'a, 'b> {
    cursor: &'a SortKeyCursor,
    sort_options: &'b [SortOptions],
}

impl<'a, 'b> SortKeyCursorWrapper {
    pub fn new(cursor: &SortKeyCursor, sort_options: &[SortOptions]) -> Self {
        Self {
            cursor,
            sort_options,
        }
    }

    pub fn compare(&mut self, other: &Self) -> Result<Ordering> {
        self.cursor.compare(other.cursor, self.sort_options)
    }
}

impl<'a, 'b> Ord for SortKeyCursorWrapper<'a, 'b> {
    fn cmp(&self, other: &mut Self) -> Ordering {
        other.compare(self).unwrap()
    }
}

impl<'a, 'b> PartialEq for SortKeyCursorWrapper<'a, 'b> {
    fn eq(&self, other: &mut Self) -> bool {
        other.compare(self).unwrap() == Ordering::Equal
    }
}

impl<'a, 'b> Eq for SortKeyCursorWrapper<'a, 'b> {}

impl<'a, 'b> PartialOrd for SortKeyCursorWrapper<'a, 'b> {
    fn partial_cmp(&self, other: &mut Self) -> Option<Ordering> {
        other.compare(self).ok()
    }
}

enum StreamWrapper {
    Receiver(mpsc::Receiver<ArrowResult<RecordBatch>>),
    Stream(Option<SendableRecordBatchStream>),
}

impl Drop for StreamWrapper {
    fn drop(&mut self) {
        match self {
            StreamWrapper::Receiver(receiver) => drop(receiver),
            StreamWrapper::Stream(stream) => {
                if let Some(s) = stream {
                    drop(s)
                }
            }
        }
    }
}

impl Stream for StreamWrapper {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self {
            StreamWrapper::Receiver(receiver) => receiver.poll_next(cx),
            StreamWrapper::Stream(ref mut stream) => {
                let inner = match stream {
                    None => return Poll::Ready(None),
                    Some(inner) => inner,
                };

                match inner.poll_next(cx) {
                    Poll::Ready(msg) => {
                        if msg.is_none() {
                            *stream = None
                        }
                        Poll::Ready(msg)
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

impl FusedStream for StreamWrapper {
    fn is_terminated(&self) -> bool {
        match self {
            StreamWrapper::Receiver(receiver) => receiver.is_terminated(),
            StreamWrapper::Stream(stream) => stream.is_none(),
        }
    }
}
