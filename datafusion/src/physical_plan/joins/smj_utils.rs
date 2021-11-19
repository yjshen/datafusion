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

use crate::arrow_dyn_list_array::DynMutableListArray;
use crate::error::{DataFusionError, Result};
use crate::execution::runtime_env::RuntimeEnv;
use crate::logical_plan::JoinType;
use crate::physical_plan::expressions::{
    exprs_to_sort_columns, Column, PhysicalSortExpr,
};
use crate::physical_plan::joins::{comp_rows, equal_rows, ColumnIndex};
use crate::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use arrow::array::*;
use arrow::array::{ArrayRef, MutableArray, MutableBooleanArray};
use arrow::compute::partition::lexicographical_partition_ranges;
use arrow::datatypes::*;
use arrow::error::ArrowError;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use futures::{Stream, StreamExt};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::iter::repeat;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Streaming Side
///////////////////////////////////////////////////////////////////////////////////////////////////

struct StreamingSideBuffer {
    batch: Option<PartitionedRecordBatch>,
    cur_row: usize,
    cur_range: usize,
    num_rows: usize,
    num_ranges: usize,
    is_new_key: bool,
    on_column: Vec<Column>,
}

impl StreamingSideBuffer {
    fn new(on_column: Vec<Column>) -> Self {
        Self {
            batch: None,
            cur_row: 0,
            cur_range: 0,
            num_rows: 0,
            num_ranges: 0,
            is_new_key: true,
            on_column,
        }
    }

    fn join_arrays(&self) -> Vec<ArrayRef> {
        join_arrays(&self.batch.unwrap().batch, &self.on_column)
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

    fn repeat_cell(
        &self,
        times: usize,
        to: &mut Box<dyn MutableArray>,
        column_index: &ColumnIndex,
    ) {
        repeat_cell(
            &self.batch.unwrap().batch,
            self.cur_row,
            times,
            to,
            column_index,
        );
    }

    fn copy_slices(
        &self,
        slice: &Slice,
        array: &mut Box<dyn MutableArray>,
        column_index: &ColumnIndex,
    ) {
        let batches = vec![&self.batch.unwrap().batch];
        let slices = vec![slice.clone()];
        copy_slices(&batches, &slices, array, column_index);
    }
}

struct StreamingSideStream {
    input: SendableRecordBatchStream,
    buffer: StreamingSideBuffer,
    input_is_finished: bool,
    sort: Vec<PhysicalSortExpr>,
}

impl StreamingSideStream {
    fn new(
        input: SendableRecordBatchStream,
        on: Vec<Column>,
        sort: Vec<PhysicalSortExpr>,
    ) -> Self {
        let buffer = StreamingSideBuffer::new(on);
        Self {
            input,
            buffer,
            input_is_finished: false,
            sort,
        }
    }

    fn input_next(
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
                        self.buffer.reset(prb);
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
        if self.buffer.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            if self.buffer.is_finished() {
                match self.input_next(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(x) => match x {
                        None => Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if self.buffer.is_finished() {
                                Poll::Ready(None)
                            } else {
                                Poll::Ready(Some(Ok(())))
                            }
                        }
                    },
                }
            } else {
                self.buffer.advance();
                Poll::Ready(Some(Ok(())))
            }
        }
    }

    fn advance_key(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.buffer.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            if self.buffer.is_finished() || self.buffer.is_last_key_in_batch() {
                match self.input_next(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(x) => match x {
                        None => Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if self.buffer.is_finished() {
                                Poll::Ready(None)
                            } else {
                                Poll::Ready(Some(Ok(())))
                            }
                        }
                    },
                }
            } else {
                self.buffer.advance_key();
                Poll::Ready(Some(Ok(())))
            }
        }
    }

    fn advance_key_skip_null(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.buffer.is_finished() && self.input_is_finished {
            Poll::Ready(None)
        } else {
            loop {
                match self.advance_key(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if !self.buffer.key_any_null() {
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Buffering Side
///////////////////////////////////////////////////////////////////////////////////////////////////

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
}

impl BufferingSideBuffer {
    fn new(on_column: Vec<Column>) -> Self {
        Self {
            batches: VecDeque::new(),
            ranges: VecDeque::new(),
            key_idx: None,
            row_num: 0,
            next_key_batch: None,
            on_column,
        }
    }

    fn join_arrays(&self) -> Vec<ArrayRef> {
        join_arrays(&self.batches[0].batch, &self.on_column)
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
        let range_len = first_range.len();
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
        self.row_num = first_range.len();
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
            self.row_num = batch.ranges[next_range_idx].len();
        }
    }

    /// Locate the starting idx for each of the ranges in the current buffer.
    fn range_start_indices(&self) -> Vec<usize> {
        let mut idx = 0;
        let mut start_indices: Vec<usize> = vec![];
        self.ranges.iter().for_each(|r| {
            start_indices.push(idx);
            idx += r.len();
        });
        start_indices.push(usize::MAX);
        start_indices
    }

    /// Locate buffered records start from `buffered_idx` of `len`gth
    fn slices_from_batches(
        &self,
        start_indices: &Vec<usize>,
        buffered_idx: usize,
        len: usize,
    ) -> Vec<Slice> {
        let ranges = &self.ranges;
        let mut idx = buffered_idx;
        let mut slices: Vec<Slice> = vec![];
        let mut remaining = len;
        let find = start_indices
            .iter()
            .enumerate()
            .find(|(i, start_idx)| **start_idx >= idx)
            .unwrap();
        let mut batch_idx = if *find.1 == idx { find.0 } else { find.0 - 1 };

        while remaining > 0 {
            let current_range = &ranges[batch_idx];
            let range_start_idx = start_indices[batch_idx];
            let start_idx = idx - range_start_idx + current_range.start;
            let range_available = current_range.len() - (idx - range_start_idx);

            if range_available >= remaining {
                slices.push(Slice {
                    batch_idx,
                    start_idx,
                    len: remaining,
                });
                remaining = 0;
            } else {
                slices.push(Slice {
                    batch_idx,
                    start_idx,
                    len: range_available,
                });
                remaining -= range_available;
                batch_idx += 1;
                idx += range_available;
            }
        }
        slices
    }

    fn copy_slices(
        &self,
        slices: &Vec<Slice>,
        array: &mut Box<dyn MutableArray>,
        column_index: &ColumnIndex,
    ) {
        let batches = self
            .batches
            .iter()
            .map(|prb| &prb.batch)
            .collect::<Vec<_>>();
        copy_slices(&batches, slices, array, column_index);
    }
}

/// Slice of batch at `batch_idx` inside BufferingSideBuffer.
#[derive(Copy, Clone)]
struct Slice {
    batch_idx: usize,
    start_idx: usize,
    len: usize,
}

struct BufferingSideStream {
    input: SendableRecordBatchStream,
    buffer: BufferingSideBuffer,
    input_is_finished: bool,
    cumulating: bool,
    sort: Vec<PhysicalSortExpr>,
}

impl BufferingSideStream {
    fn new(
        input: SendableRecordBatchStream,
        on: Vec<Column>,
        sort: Vec<PhysicalSortExpr>,
    ) -> Self {
        let buffer = BufferingSideBuffer::new(on);
        Self {
            input,
            buffer,
            input_is_finished: false,
            cumulating: false,
            sort,
        }
    }

    fn input_next(
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
        if self.buffer.is_finished()? && self.input_is_finished {
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

            if self.buffer.is_finished()? {
                return match &self.buffer.next_key_batch {
                    None => match self.input_next(cx) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(x) => match x {
                            None => Poll::Ready(None),
                            Some(x) => {
                                let prb = x?;
                                self.buffer.reset(&prb);
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
                        self.buffer.reset(batch);
                        if batch.ranges.len() == 1 {
                            self.cumulating = true;
                            Poll::Pending
                        } else {
                            Poll::Ready(Some(Ok(())))
                        }
                    }
                };
            } else {
                self.buffer.advance_key();
                if self.buffer.batches[0].is_last_range(&self.buffer.ranges[0]) {
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
        if self.buffer.is_finished()? && self.input_is_finished {
            Poll::Ready(None)
        } else {
            loop {
                match self.advance_key(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(x) => {
                            x?;
                            if !self.buffer.key_any_null() {
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
            match self.input_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(x) => match x {
                    None => {
                        self.cumulating = false;
                        return Poll::Ready(Some(Ok(())));
                    }
                    Some(x) => {
                        let prb = x?;
                        let buffer_more = self.buffer.running_key(&prb)?;
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Output
///////////////////////////////////////////////////////////////////////////////////////////////////

struct OutputBuffer {
    arrays: Vec<Box<dyn MutableArray>>,
    target_batch_size: usize,
    slots_available: usize,
    schema: Arc<Schema>,
}

impl OutputBuffer {
    fn new(target_batch_size: usize, schema: Arc<Schema>) -> ArrowResult<Self> {
        let arrays = new_arrays(&schema, target_batch_size)?;
        Ok(Self {
            arrays,
            target_batch_size,
            slots_available: target_batch_size,
            schema,
        })
    }

    fn output_and_reset(mut self) -> ArrowResult<RecordBatch> {
        let result = make_batch(self.schema.clone(), self.arrays);
        self.arrays = new_arrays(&self.schema, self.target_batch_size)?;
        self.slots_available = self.target_batch_size;
        result
    }

    fn append(&mut self, size: usize) {
        assert!(size <= self.slots_available);
        self.slots_available -= size;
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.slots_available == 0
    }
}

fn new_arrays(
    schema: &Arc<Schema>,
    batch_size: usize,
) -> ArrowResult<Vec<Box<dyn MutableArray>>> {
    let arrays: Vec<Box<dyn MutableArray>> = schema
        .fields()
        .iter()
        .map(|field| {
            let dt = field.data_type.to_logical_type();
            make_mutable(dt, batch_size)
        })
        .collect::<ArrowResult<_>>()?;
    Ok(arrays)
}

fn make_batch(
    schema: Arc<Schema>,
    mut arrays: Vec<Box<dyn MutableArray>>,
) -> ArrowResult<RecordBatch> {
    let columns = arrays.iter_mut().map(|array| array.as_arc()).collect();
    RecordBatch::try_new(schema, columns)
}

macro_rules! with_match_primitive_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use arrow::datatypes::PrimitiveType::*;
    use arrow::types::{days_ms, months_days_ns};
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        DaysMs => __with_ty__! { days_ms },
        MonthDayNano => __with_ty__! { months_days_ns },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
    }
})}

fn make_mutable(
    data_type: &DataType,
    capacity: usize,
) -> ArrowResult<Box<dyn MutableArray>> {
    Ok(match data_type.to_physical_type() {
        PhysicalType::Boolean => Box::new(MutableBooleanArray::with_capacity(capacity))
            as Box<dyn MutableArray>,
        PhysicalType::Primitive(primitive) => {
            with_match_primitive_type!(primitive, |$T| {
                Box::new(MutablePrimitiveArray::<$T>::with_capacity(capacity).to(data_type.clone()))
                    as Box<dyn MutableArray>
            })
        }
        PhysicalType::Binary => {
            Box::new(MutableBinaryArray::<i32>::with_capacity(capacity))
                as Box<dyn MutableArray>
        }
        PhysicalType::Utf8 => Box::new(MutableUtf8Array::<i32>::with_capacity(capacity))
            as Box<dyn MutableArray>,
        _ => match data_type {
            DataType::List(inner) => {
                let values = make_mutable(inner.data_type(), 0)?;
                Box::new(DynMutableListArray::<i32>::new_with_capacity(
                    values, capacity,
                )) as Box<dyn MutableArray>
            }
            DataType::FixedSizeBinary(size) => Box::new(
                MutableFixedSizeBinaryArray::with_capacity(*size as usize, capacity),
            ) as Box<dyn MutableArray>,
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "making mutable of type {} is not implemented yet",
                    data_type
                )))
            }
        },
    })
}

macro_rules! repeat_n {
    ($TO:ty, $FROM:ty, $N:expr, $to: ident, $from: ident, $idx: ident) => {{
        let to = $to.as_mut_any().downcast_mut::<$TO>().unwrap();
        let from = $from.as_any().downcast_ref::<$FROM>().unwrap();
        let repeat_iter = from
            .slice($idx, 1)
            .iter()
            .flat_map(|v| repeat(v).take($N))
            .collect::<Vec<_>>();
        to.extend_trusted_len(repeat_iter.into_iter());
    }};
}

/// repeat times of cell located by `idx` at streamed side to output
fn repeat_cell(
    batch: &RecordBatch,
    idx: usize,
    times: usize,
    to: &mut Box<dyn MutableArray>,
    column_index: &ColumnIndex,
) {
    let from = batch.column(column_index.index);
    match to.data_type().to_physical_type() {
        PhysicalType::Boolean => {
            repeat_n!(MutableBooleanArray, BooleanArray, times, to, from, idx)
        }
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => repeat_n!(Int8Vec, Int8Array, times, to, from, idx),
            PrimitiveType::Int16 => repeat_n!(Int16Vec, Int16Array, times, to, from, idx),
            PrimitiveType::Int32 => repeat_n!(Int32Vec, Int32Array, times, to, from, idx),
            PrimitiveType::Int64 => repeat_n!(Int64Vec, Int64Array, times, to, from, idx),
            PrimitiveType::Float32 => {
                repeat_n!(Float32Vec, Float32Array, times, to, from, idx)
            }
            PrimitiveType::Float64 => {
                repeat_n!(Float64Vec, Float64Array, times, to, from, idx)
            }
            _ => todo!(),
        },
        PhysicalType::Utf8 => {
            repeat_n!(MutableUtf8Array<i32>, Utf8Array<i32>, times, to, from, idx)
        }
        _ => todo!(),
    }
}

macro_rules! copy_slices {
    ($TO:ty, $FROM:ty, $array: ident, $batches: ident, $slices: ident, $column_index: ident) => {{
        let to = $array.as_mut_any().downcast_mut::<$TO>().unwrap();
        for pos in $slices {
            let from = $batches[pos.batch_idx]
                .column($column_index.index)
                .slice(pos.start_idx, pos.len);
            let from = from.as_any().downcast_ref::<$FROM>().unwrap();
            to.extend_trusted_len(from.iter());
        }
    }};
}

fn copy_slices(
    batches: &Vec<&RecordBatch>,
    slices: &Vec<Slice>,
    array: &mut Box<dyn MutableArray>,
    column_index: &ColumnIndex,
) {
    // output buffered start `buffered_idx`, len `rows_to_output`
    match array.data_type().to_physical_type() {
        PhysicalType::Boolean => {
            copy_slices!(
                MutableBooleanArray,
                BooleanArray,
                array,
                batches,
                slices,
                column_index
            )
        }
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => {
                copy_slices!(Int8Vec, Int8Array, array, batches, slices, column_index)
            }
            PrimitiveType::Int16 => {
                copy_slices!(Int16Vec, Int16Array, array, batches, slices, column_index)
            }
            PrimitiveType::Int32 => {
                copy_slices!(Int32Vec, Int32Array, array, batches, slices, column_index)
            }
            PrimitiveType::Int64 => {
                copy_slices!(Int64Vec, Int64Array, array, batches, slices, column_index)
            }
            PrimitiveType::Float32 => copy_slices!(
                Float32Vec,
                Float32Array,
                array,
                batches,
                slices,
                column_index
            ),
            PrimitiveType::Float64 => copy_slices!(
                Float64Vec,
                Float64Array,
                array,
                batches,
                slices,
                column_index
            ),
            _ => todo!(),
        },
        PhysicalType::Utf8 => {
            copy_slices!(
                MutableUtf8Array<i32>,
                Utf8Array<i32>,
                array,
                batches,
                slices,
                column_index
            )
        }
        _ => todo!(),
    }
}

pub struct SortMergeJoinCommon {
    streamed: StreamingSideStream,
    buffered: BufferingSideStream,
    schema: Arc<Schema>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    join_type: JoinType,
    runtime: Arc<RuntimeEnv>,
    output: OutputBuffer,
}

impl SortMergeJoinCommon {
    pub fn new(
        streamed: SendableRecordBatchStream,
        buffered: SendableRecordBatchStream,
        on_streamed: Vec<Column>,
        on_buffered: Vec<Column>,
        streamed_sort: Vec<PhysicalSortExpr>,
        buffered_sort: Vec<PhysicalSortExpr>,
        column_indices: Vec<ColumnIndex>,
        schema: Arc<Schema>,
        join_type: JoinType,
        runtime: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        let streamed = StreamingSideStream::new(streamed, on_streamed, streamed_sort);
        let buffered = BufferingSideStream::new(buffered, on_buffered, buffered_sort);
        let output = OutputBuffer::new(runtime.batch_size(), schema.clone())
            .map_err(DataFusionError::ArrowError)?;
        Ok(Self {
            streamed,
            buffered,
            schema,
            column_indices,
            join_type,
            runtime,
            output,
        })
    }

    fn compare_stream_buffer(&self) -> Result<Ordering> {
        let stream_arrays = &self.streamed.buffer.join_arrays();
        let buffer_arrays = &self.buffered.buffer.join_arrays();
        comp_rows(
            self.streamed.buffer.cur_row,
            self.buffered.buffer.key_idx.unwrap(),
            stream_arrays,
            buffer_arrays,
        )
    }
}

pub struct InnerJoiner {
    inner: SortMergeJoinCommon,
    matched: bool,
    buffered_idx: usize,
    start_indices: Vec<usize>,
    buffer_remaining: usize,
    advance_stream: bool,
    advance_buffer: bool,
    continues_match: bool,
}

impl InnerJoiner {
    pub fn new(inner: SortMergeJoinCommon) -> Self {
        Self {
            inner,
            matched: false,
            buffered_idx: 0,
            start_indices: vec![],
            buffer_remaining: 0,
            advance_stream: true,
            advance_buffer: true,
            continues_match: false,
        }
    }

    fn find_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<()>>> {
        if self.continues_match {
            match Pin::new(&mut self.inner.streamed).advance(cx) {
                Poll::Ready(x) => match x {
                    None => Poll::Ready(None),
                    Some(y) => {
                        y?;
                        if self.inner.streamed.buffer.is_new_key {
                            self.continues_match = false;
                            self.advance_stream = false;
                            self.advance_buffer = true;
                            self.matched = false;
                            self.find_next(cx)
                        } else {
                            self.continues_match = true;
                            self.advance_stream = false;
                            self.advance_buffer = false;
                            self.matched = false;
                            Poll::Ready(Some(Ok(())))
                        }
                    }
                },
                Poll::Pending => Poll::Pending,
            }
        } else {
            if self.advance_stream {
                match Pin::new(&mut self.inner.streamed).advance_key_skip_null(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(y) => {
                            y?;
                            self.continues_match = true;
                        }
                    },
                    Poll::Pending => return Poll::Pending,
                }
            }

            if self.advance_buffer {
                match Pin::new(&mut self.inner.streamed).advance_key_skip_null(cx) {
                    Poll::Ready(x) => match x {
                        None => return Poll::Ready(None),
                        Some(y) => {
                            y?;
                        }
                    },
                    Poll::Pending => return Poll::Pending,
                }
            }

            let cmp = self
                .inner
                .compare_stream_buffer()
                .map_err(DataFusionError::into_arrow_external_error)?;

            match cmp {
                Ordering::Less => {
                    self.advance_stream = true;
                    self.advance_buffer = false;
                    self.find_next(cx)
                }
                Ordering::Equal => {
                    self.advance_stream = true;
                    self.advance_buffer = true;
                    self.matched = true;
                    self.buffered_idx = 0;
                    self.start_indices = self.inner.buffered.buffer.range_start_indices();
                    self.buffer_remaining = self.inner.buffered.buffer.row_num;
                    Poll::Ready(Some(Ok(())))
                }
                Ordering::Greater => {
                    self.advance_stream = false;
                    self.advance_buffer = true;
                    self.find_next(cx)
                }
            }
        }
    }

    fn fill_output(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ArrowResult<RecordBatch>>> {
        let output = &mut self.inner.output;
        let streamed = &self.inner.streamed.buffer;
        let buffered = &self.inner.buffered.buffer;

        let slots_available = output.slots_available;
        let mut rows_to_output = 0;
        if slots_available >= self.buffer_remaining {
            self.matched = false;
            rows_to_output = self.buffer_remaining;
            self.buffer_remaining = 0;
        } else {
            rows_to_output = slots_available;
            self.buffer_remaining -= rows_to_output;
        }

        let slices = buffered.slices_from_batches(
            &self.start_indices,
            self.buffered_idx,
            rows_to_output,
        );

        output
            .arrays
            .iter_mut()
            .zip(self.inner.schema.fields().iter())
            .zip(self.inner.column_indices.iter())
            .map(|((array, field), column_index)| {
                if column_index.is_left {
                    // repeat streamed `rows_to_output` times
                    streamed.repeat_cell(rows_to_output, array, column_index);
                } else {
                    // copy buffered start from: `buffered_idx`, len: `rows_to_output`
                    buffered.copy_slices(&slices, array, column_index);
                }
            });

        self.inner.output.append(rows_to_output);
        self.buffered_idx += rows_to_output;

        if self.inner.output.is_full() {
            let result = output.output_and_reset();
            Poll::Ready(Some(result))
        } else {
            self.poll_next(cx)
        }
    }
}

impl Stream for InnerJoiner {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.matched {
            let find = self.find_next(cx);
            match find {
                Poll::Ready(x) => match x {
                    None => Poll::Ready(None),
                    Some(y) => match y {
                        Ok(_) => self.fill_output(cx),
                        Err(err) => Poll::Ready(Some(Err(ArrowError::External(
                            "Failed while finding next match for inner join".to_owned(),
                            err.into(),
                        )))),
                    },
                },
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.fill_output(cx)
        }
    }
}

impl RecordBatchStream for InnerJoiner {
    fn schema(&self) -> SchemaRef {
        self.inner.schema.clone()
    }
}
