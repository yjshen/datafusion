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

//! Defines the join plan for executing partitions in parallel and then joining the results
//! into a set of partitions.

use std::fmt;
use std::iter::repeat;
use std::sync::Arc;
use std::{any::Any, usize};
use std::{time::Instant, vec};

use arrow::compute::take;
use arrow::datatypes::*;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;
use arrow::{array::*, buffer::MutableBuffer};
use async_trait::async_trait;
use futures::{Stream, StreamExt, TryStreamExt};
use log::debug;
use smallvec::{smallvec, SmallVec};
use tokio::sync::Mutex;

use crate::error::{DataFusionError, Result};
use crate::execution::runtime_env::RuntimeEnv;
use crate::logical_plan::JoinType;
use crate::physical_plan::coalesce_batches::concat_batches;
use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use crate::physical_plan::expressions::{exprs_to_sort_columns, PhysicalSortExpr};
use crate::physical_plan::joins::{
    build_join_schema, check_join_is_valid, column_indices_from_schema, comp_rows,
    equal_rows, JoinOn,
};
use crate::physical_plan::sorts::external_sort::ExternalSortExec;
use crate::physical_plan::stream::RecordBatchReceiverStream;
use crate::physical_plan::PhysicalExpr;
use crate::physical_plan::{
    expressions::Column,
    metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
};
use crate::physical_plan::{hash_utils::create_hashes, Statistics};
use crate::physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
    SendableRecordBatchStream,
};
use arrow::array::growable::GrowablePrimitive;
use arrow::compute::partition::lexicographical_partition_ranges;
use arrow::compute::sort::SortOptions;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::Range;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc::{Receiver, Sender};

type StringArray = Utf8Array<i32>;
type LargeStringArray = Utf8Array<i64>;

#[derive(Clone)]
struct PartitionedRecordBatch {
    batch: RecordBatch,
    ranges: Vec<Range<usize>>,
}

impl PartitionedRecordBatch {
    fn new(
        batch: Option<RecordBatch>,
        expr: &[PhysicalSortExpr],
    ) -> Result<Option<Self>> {
        match batch {
            Some(batch) => {
                let columns = exprs_to_sort_columns(&batch, expr)?;
                let ranges =
                    lexicographical_partition_ranges(&columns)?.collect::<Vec<_>>();
                Ok(Some(Self { batch, ranges }))
            }
            None => Ok(None),
        }
    }

    #[inline]
    fn is_last_range(&self, range: &Range<usize>) -> bool {
        range.end == self.batch.num_rows()
    }
}

struct StreamingBatch {
    batch: Option<PartitionedRecordBatch>,
    cur_row: usize,
    cur_range: usize,
    num_rows: usize,
    num_ranges: uszie,
    is_new_key: bool,
    on_column: Vec<Column>,
    sort: Vec<PhysicalSortExpr>,
}

impl StreamingBatch {
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

    fn rest_batch(&mut self, prb: Option<PartitionedRecordBatch>) {
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
        } else {
            self.batch = None;
        }
    }

    fn advance_key(&mut self) {
        let ranges = self.batch.unwrap().ranges;
        self.cur_range += 1;
        self.cur_row = ranges[self.cur_range].start;
        self.is_new_key = true;
    }
}

/// Holding ranges for same key over several bathes
struct BufferedBatches {
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
fn range_len(range: &Range<usize>) -> usize {
    range.end - range.start
}

impl BufferedBatches {
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

    fn is_finished(&self) -> Result<bool> {
        match self.key_idx {
            None => Ok(true),
            Some(_) => match (self.batches.back(), self.ranges.back()) {
                (Some(batch), Some(range)) => Ok(batch.is_last_range(range)),
                _ => Err(DataFusionError::Execution(format!(
                    "Batches length {} not equal to ranges length {}",
                    self.batches.len(),
                    self.ranges.len()
                ))),
            },
        }
    }

    /// Whether the running key ends at the current batch `prb`, true for continues, false for ends.
    fn running_key(&mut self, prb: &PartitionedRecordBatch) -> Result<bool> {
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
                let equal = equal_rows(key_idx, 0, &key_arrays, &current_arrays)?;
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

    fn reset_batch(&mut self, prb: &PartitionedRecordBatch) {
        self.cleanup();
        self.batches.push_back(prb.clone());
        let first_range = &prb.ranges[0];
        self.ranges.push_back(first_range.clone());
        self.key_idx = Some(0);
        self.row_num = range_len(first_range);
    }

    /// Advance the cursor to the next key seen by this buffer
    fn advance_in_current_batch(&mut self) {
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
                .find_position(|x| x.start == tail_range.start)
                .unwrap()
                .0;
            self.key_idx = Some(tail_range.end);
            self.ranges.push_back(batch.ranges[next_range_idx].clone());
            self.row_num = range_len(&batch.ranges[next_range_idx]);
        }
    }
}

fn join_arrays(rb: &RecordBatch, on_column: &Vec<Column>) -> Vec<ArrayRef> {
    on_column.iter().map(|c| rb.column(c.index())).collect()
}

struct SortMergeJoinDriver {
    streamed: SendableRecordBatchStream,
    buffered: SendableRecordBatchStream,
    on_streamed: Vec<Column>,
    on_buffered: Vec<Column>,
    schema: Arc<Schema>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    stream_batch: StreamingBatch,
    buffered_batches: BufferedBatches,
    runtime: Arc<RuntimeEnv>,
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

fn make_mutable(data_type: &DataType, capacity: usize) -> Result<Box<dyn MutableArray>> {
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
                return Err(DataFusionError::Execution(format!(
                    "making mutable of type {} is not implemented yet",
                    data_type
                )))
            }
        },
    })
}

fn new_arrays(
    schema: &Arc<Schema>,
    batch_size: usize,
) -> Result<Vec<Box<dyn MutableArray>>> {
    let arrays: Vec<Box<dyn MutableArray>> = schema
        .fields()
        .iter()
        .map(|field| {
            let dt = field.data_type.to_logical_type();
            make_mutable(dt, batch_size)
        })
        .collect::<Result<_>>()?;
    Ok(arrays)
}

fn make_batch(
    schema: Arc<Schema>,
    mut arrays: Vec<Box<dyn MutableArray>>,
) -> ArrowResult<RecordBatch> {
    let columns = arrays.iter_mut().map(|array| array.as_arc()).collect();
    RecordBatch::try_new(schema, columns)
}

macro_rules! repeat_n {
    ($TO:ty, $FROM:ty, $N:expr) => {{
        let to = to.as_mut_any().downcast_mut::<$TO>().unwrap();
        let from = from.as_any().downcast_ref::<$FROM>().unwrap();
        let repeat_iter = from.slice(idx, 1).iter().flat_map(|v| repeat(v).take($N));
        to.extend_trusted_len(repeat_iter);
    }};
}

/// repeat times of cell located by `idx` at streamed side to output
fn repeat_streamed_cell(
    stream_batch: &RecordBatch,
    idx: usize,
    times: usize,
    to: &mut Box<dyn MutableArray>,
    column_index: &ColumnIndex,
) {
    let from = stream_batch.column(column_index.index);
    match to.data_type().to_physical_type() {
        PhysicalType::Boolean => {
            repeat_n!(MutableBooleanArray, BooleanArray, times)
        }
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => repeat_n!(Int8Vec, Int8Array, times),
            PrimitiveType::Int16 => repeat_n!(Int16Vec, Int16Array, times),
            PrimitiveType::Int32 => repeat_n!(Int32Vec, Int32Array, times),
            PrimitiveType::Int64 => repeat_n!(Int64Vec, Int64Array, times),
            PrimitiveType::Float32 => repeat_n!(Float32Vec, Float32Array, times),
            PrimitiveType::Float64 => repeat_n!(Float64Vec, Float64Array, times),
            _ => todo!(),
        },
        PhysicalType::Utf8 => {
            repeat_n!(MutableUtf8Array<i32>, Utf8Array<i32>, times)
        }
        PhysicalType::Binary => {
            repeat_n!(MutableBinaryArray<i32>, BinaryArray<i32>, times)
        }
        PhysicalType::FixedSizeBinary => {
            repeat_n!(MutableFixedSizeBinaryArray, FixedSizeBinaryArray, times)
        }
        _ => todo!(),
    }
}

macro_rules! copy_slices {
    ($TO:ty, $FROM:ty) => {{
        let to = array.as_mut_any().downcast_mut::<$TO>().unwrap();
        for pos in slices {
            let from = buffered_batches[pos.batch_idx]
                .batch
                .column(column_index.index)
                .slice(pos.start_idx, pos.len);
            let from = from.as_any().downcast_ref::<$FROM>().unwrap();
            to.extend_trusted_len(from.iter());
        }
    }};
}

fn copy_buffered_slices(
    buffered_batches: &VecDeque<PartitionedRecordBatch>,
    slices: &Vec<Slice>,
    array: &mut Box<dyn MutableArray>,
    column_index: &ColumnIndex,
) {
    // output buffered start `buffered_idx`, len `rows_to_output`
    match to.data_type().to_physical_type() {
        PhysicalType::Boolean => {
            copy_slices!(MutableBooleanArray, BooleanArray)
        }
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => copy_slices!(Int8Vec, Int8Array),
            PrimitiveType::Int16 => copy_slices!(Int16Vec, Int16Array),
            PrimitiveType::Int32 => copy_slices!(Int32Vec, Int32Array),
            PrimitiveType::Int64 => copy_slices!(Int64Vec, Int64Array),
            PrimitiveType::Float32 => copy_slices!(Float32Vec, Float32Array),
            PrimitiveType::Float64 => copy_slices!(Float64Vec, Float64Array),
            _ => todo!(),
        },
        PhysicalType::Utf8 => {
            copy_slices!(MutableUtf8Array<i32>, Utf8Array<i32>)
        }
        PhysicalType::Binary => {
            copy_slices!(MutableBinaryArray<i32>, BinaryArray<i32>)
        }
        PhysicalType::FixedSizeBinary => {
            copy_slices!(MutableFixedSizeBinaryArray, FixedSizeBinaryArray)
        }
        _ => todo!(),
    }
}

fn range_start_indices(buffered_ranges: &VecDeque<Range<usize>>) -> Vec<usize> {
    let mut idx = 0;
    let mut start_indices: Vec<usize> = vec![];
    buffered_ranges
        .iter()
        .for_each(|r| {
            start_indices.push(idx);
            idx += range_len(r);
        })
        .collect();
    start_indices.push(usize::MAX);
    start_indices
}

/// Locate buffered records start from `buffered_idx` of `len`gth
/// inside buffered batches.
fn slices_from_batches(
    buffered_ranges: &VecDeque<Range<usize>>,
    start_indices: &Vec<usize>,
    buffered_idx: usize,
    len: usize,
) -> Vec<Slice> {
    let mut idx = buffered_idx;
    let mut slices: Vec<Slice> = vec![];
    let mut remaining = len;
    let find = start_indices
        .iter()
        .find_position(|&&start_idx| start_idx >= idx)
        .unwrap();
    let mut batch_idx = if find.1 == idx { find.0 } else { find.0 - 1 };

    while remaining > 0 {
        let current_range = &buffered_ranges[batch_idx];
        let range_start_idx = start_indices[batch_idx];
        let start_idx = idx - range_start_idx + current_range.start;
        let range_available = range_len(current_range) - (idx - range_start_idx);

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

/// Slice of batch at `batch_idx` inside BufferedBatches.
struct Slice {
    batch_idx: usize,
    start_idx: usize,
    len: usize,
}

impl SortMergeJoinDriver {
    fn new(
        streamed: SendableRecordBatchStream,
        buffered: SendableRecordBatchStream,
        on_streamed: Vec<Column>,
        on_buffered: Vec<Column>,
        streamed_sort: Vec<PhysicalSortExpr>,
        buffered_sort: Vec<PhysicalSortExpr>,
        column_indices: Vec<ColumnIndex>,
        schema: Arc<Schema>,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        Self {
            streamed,
            buffered,
            on_streamed,
            on_buffered,
            schema,
            column_indices,
            stream_batch: StreamingBatch::new(on_streamed.clone(), streamed_sort),
            buffered_batches: BufferedBatches::new(on_buffered.clone(), buffered_sort),
            runtime,
        }
    }

    async fn inner_join_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let target_batch_size = self.runtime.batch_size();

        let mut output_slots_available = target_batch_size;
        let mut output_arrays = new_arrays(&self.schema, target_batch_size)?;

        while self.find_next_inner_match()? {
            loop {
                let result = self
                    .join_eq_records(
                        target_batch_size,
                        output_slots_available,
                        output_arrays,
                        sender,
                    )
                    .await?;
                output_slots_available = result.0;
                output_arrays = result.1;

                self.stream_batch.advance();
                if self.stream_batch.is_new_key {
                    break;
                }
            }
        }

        Ok(())
    }

    async fn join_eq_records(
        &mut self,
        target_batch_size: usize,
        output_slots_available: usize,
        mut output_arrays: Vec<Box<dyn MutableArray>>,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<(usize, Vec<Box<dyn MutableArray>>)> {
        let mut output_slots_available = output_slots_available;
        let mut remaining = self.buffered_batches.row_num;
        let stream_batch = &self.stream_batch.batch.unwrap().batch;
        let stream_row = self.stream_batch.cur_row;

        let buffered_batches = &self.buffered_batches.batches;
        let buffered_ranges = &self.buffered_batches.ranges;

        let mut unfinished = true;
        let mut buffered_idx = 0;
        let mut rows_to_output = 0;
        let start_indices = range_start_indices(buffered_ranges);

        // output each buffered matching record once
        while unfinished {
            if output_slots_available >= remaining {
                unfinished = false;
                rows_to_output = remaining;
                output_slots_available -= remaining;
                remaining = 0;
            } else {
                rows_to_output = output_slots_available;
                output_slots_available = 0;
                remaining -= rows_to_output;
            }

            // get slices for buffered side for the current output
            let slices = slices_from_batches(
                buffered_ranges,
                &start_indices,
                buffered_idx,
                rows_to_output,
            );

            output_arrays
                .iter_mut()
                .zip(self.schema.fields().iter())
                .zip(self.column_indices.iter())
                .map(|((array, field), column_index)| {
                    if column_index.is_left {
                        // repeat streamed `rows_to_output` times
                        repeat_streamed_cell(
                            stream_batch,
                            stream_row,
                            rows_to_output,
                            array,
                            column_index,
                        );
                    } else {
                        // copy buffered start from: `buffered_idx`, len: `rows_to_output`
                        copy_buffered_slices(
                            buffered_batches,
                            &slices,
                            array,
                            column_index,
                        )
                    }
                });

            if output_slots_available == 0 {
                let result = make_batch(self.schema.clone(), output_arrays);

                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via inner join stream: {}", e);
                };

                output_arrays = new_arrays(&self.schema, target_batch_size)?;
                output_slots_available = target_batch_size;
            }

            buffered_idx += rows_to_output;
            rows_to_output = 0;
        }
        Ok((output_slots_available, output_arrays))
    }

    fn find_next_inner_match(&mut self) -> Result<bool> {
        if self.stream_batch.key_any_null() {
            let more_stream = self.advance_streamed_key_null_free()?;
            if !more_stream {
                return Ok(false);
            }
        }

        if self.buffered_batches.key_any_null() {
            let more_buffer = self.advance_buffered_key_null_free()?;
            if !more_buffer {
                return Ok(false);
            }
        }

        loop {
            let current_cmp = self.compare_stream_buffer()?;
            match current_cmp {
                Ordering::Less => {
                    let more_stream = self.advance_streamed_key_null_free()?;
                    if !more_stream {
                        return Ok(false);
                    }
                }
                Ordering::Equal => return Ok(true),
                Ordering::Greater => {
                    let more_buffer = self.advance_buffered_key_null_free()?;
                    if !more_buffer {
                        return Ok(false);
                    }
                }
            }
        }
    }

    /// true for has next, false for ended
    fn advance_streamed(&mut self) -> Result<bool> {
        if self.stream_batch.is_finished() {
            self.get_stream_next()?;
            Ok(!self.stream_batch.is_finished())
        } else {
            self.stream_batch.advance();
            Ok(true)
        }
    }

    /// true for has next, false for ended
    fn advance_streamed_key(&mut self) -> Result<bool> {
        if self.stream_batch.is_finished() || self.stream_batch.is_last_key_in_batch() {
            self.get_stream_next()?;
            Ok(!self.stream_batch.is_finished())
        } else {
            self.stream_batch.advance_key();
            Ok(true)
        }
    }

    /// true for has next, false for ended
    fn advance_streamed_key_null_free(&mut self) -> Result<bool> {
        let mut more_stream_keys = self.advance_streamed_key()?;
        loop {
            if more_stream_keys && self.stream_batch.key_any_null() {
                more_stream_keys = self.advance_streamed_key()?;
            } else {
                break;
            }
        }
        Ok(more_stream_keys)
    }

    fn advance_buffered_key_null_free(&mut self) -> Result<bool> {
        let mut more_buffered_keys = self.advance_buffered_key()?;
        loop {
            if more_buffered_keys && self.buffered_batches.key_any_null() {
                more_buffered_keys = self.advance_buffered_key()?;
            } else {
                break;
            }
        }
        Ok(more_buffered_keys)
    }

    /// true for has next, false for ended
    fn advance_buffered_key(&mut self) -> Result<bool> {
        if self.buffered_batches.is_finished() {
            match &self.buffered_batches.next_key_batch {
                None => {
                    let batch = self.get_buffered_next()?;
                    match batch {
                        None => return Ok(false),
                        Some(batch) => {
                            self.buffered_batches.reset_batch(&batch);
                        }
                    }
                }
                Some(batch) => {
                    self.buffered_batches.reset_batch(batch);
                }
            }
        } else {
            self.buffered_batches.advance_in_current_batch();
        }
        Ok(false)
    }

    /// true for has next, false for buffer side ended
    fn cumulate_same_keys(&mut self) -> Result<bool> {
        let batch = self.get_buffered_next()?;
        match batch {
            None => Ok(false),
            Some(batch) => {
                let more_batches = self.buffered_batches.running_key(&batch)?;
                if more_batches {
                    self.cumulate_same_keys()
                } else {
                    // reach end of current key, but the stream continues
                    Ok(true)
                }
            }
        }
    }

    fn compare_stream_buffer(&self) -> Result<Ordering> {
        let stream_arrays =
            join_arrays(&self.stream_batch.batch.unwrap().batch, &self.on_streamed);
        let buffer_arrays =
            join_arrays(&self.buffered_batches.batches[0].batch, &self.on_buffered);
        comp_rows(
            self.stream_batch.cur_row,
            self.buffered_batches.key_idx.unwrap(),
            &stream_arrays,
            &buffer_arrays,
        )
    }

    fn get_stream_next(&mut self) -> Result<()> {
        let batch = self.streamed.next().await.transpose()?;
        let prb = PartitionedRecordBatch::new(batch, &self.stream_batch.sort)?;
        self.stream_batch.rest_batch(prb);
        Ok(())
    }

    fn get_buffered_next(&mut self) -> Result<Option<PartitionedRecordBatch>> {
        let batch = self.buffered.next().await.transpose()?;
        PartitionedRecordBatch::new(batch, &self.buffered_batches.sort)
    }
}

/// join execution plan executes partitions in parallel and combines them into a set of
/// partitions.
#[derive(Debug)]
pub struct SortMergeJoinExec {
    /// left (build) side which gets hashed
    left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    right: Arc<dyn ExecutionPlan>,
    /// Set of common columns used to join on
    on: Vec<(Column, Column)>,
    /// How the join is performed
    join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

/// Metrics for SortMergeJoinExec
#[derive(Debug)]
struct SortMergeJoinMetrics {
    /// Total time for joining probe-side batches to the build-side batches
    join_time: metrics::Time,
    /// Number of batches consumed by this operator
    input_batches: metrics::Count,
    /// Number of rows consumed by this operator
    input_rows: metrics::Count,
    /// Number of batches produced by this operator
    output_batches: metrics::Count,
    /// Number of rows produced by this operator
    output_rows: metrics::Count,
}

impl SortMergeJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            join_time,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

/// Information about the index and placement (left or right) of the columns
struct ColumnIndex {
    /// Index of the column
    index: usize,
    /// Whether the column is at the left or right side
    is_left: bool,
}

impl SortMergeJoinExec {
    /// Tries to create a new [SortMergeJoinExec].
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &on)?;

        let schema = Arc::new(build_join_schema(&left_schema, &right_schema, join_type));

        Ok(SortMergeJoinExec {
            left,
            right,
            on,
            join_type: *join_type,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// left (build) side which gets hashed
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(Column, Column)] {
        &self.on
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }
}

#[async_trait]
impl ExecutionPlan for SortMergeJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            2 => Ok(Arc::new(SortMergeJoinExec::try_new(
                children[0].clone(),
                children[1].clone(),
                self.on.clone(),
                &self.join_type,
            )?)),
            _ => Err(DataFusionError::Internal(
                "HashJoinExec wrong number of children".to_string(),
            )),
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        self.right.output_partitioning()
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        let on_left = self.on.iter().map(|on| on.0.clone()).collect::<Vec<_>>();
        let on_right = self.on.iter().map(|on| on.1.clone()).collect::<Vec<_>>();
        let left = self.left.execute(partition).await?;
        let right = self.right.execute(partition).await?;

        let column_indices = column_indices_from_schema(
            &self.join_type,
            &self.left.schema(),
            &self.right.schema(),
            &self.schema,
        )?;

        let (tx, rx): (
            Sender<ArrowResult<RecordBatch>>,
            Receiver<ArrowResult<RecordBatch>>,
        ) = tokio::sync::mpsc::channel(2);

        let left_sort = self
            .left
            .as_any()
            .downcast_ref::<ExternalSortExec>()
            .unwrap()
            .expr()
            .iter()
            .map(|s| s.clone())
            .collect::<Vec<_>>();
        let right_sort = self
            .right
            .as_any()
            .downcast_ref::<ExternalSortExec>()
            .unwrap()
            .expr()
            .iter()
            .map(|s| s.clone())
            .collect::<Vec<_>>();

        let mut driver = match self.join_type {
            JoinType::Inner
            | JoinType::Left
            | JoinType::Full
            | JoinType::Semi
            | JoinType::Anti => SortMergeJoinDriver::new(
                left,
                right,
                on_left,
                on_right,
                left_sort,
                right_sort,
                column_indices,
                self.schema.clone(),
                RUNTIME_ENV.clone(),
            ),
            JoinType::Right => SortMergeJoinDriver::new(
                right,
                left,
                on_right,
                on_left,
                right_sort,
                left_sort,
                column_indices,
                self.schema.clone(),
                RUNTIME_ENV.clone(),
            ),
        };

        match self.join_type {
            JoinType::Inner => driver.inner_join_driver(&tx).await?,
            JoinType::Left => {}
            JoinType::Right => {}
            JoinType::Full => {}
            JoinType::Semi => {}
            JoinType::Anti => {}
        }

        let result = RecordBatchReceiverStream::create(&schema, rx);

        Ok(Box::pin(result))
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "SortMergeJoinExec: join_type={:?}, on={:?}",
                    self.join_type, self.on
                )
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        // TODO stats: it is not possible in general to know the output size of joins
        // There are some special cases though, for example:
        // - `A LEFT JOIN B ON A.col=B.col` with `COUNT_DISTINCT(B.col)=COUNT(B.col)`
        Statistics::default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        assert_batches_sorted_eq,
        physical_plan::{
            common, expressions::Column, memory::MemoryExec, repartition::RepartitionExec,
        },
        test::{build_table_i32, columns},
    };

    use super::*;

    fn build_table(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema().clone();
        Arc::new(MemoryExec::try_new(&[vec![batch]], schema, None).unwrap())
    }

    fn join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
    ) -> Result<SortMergeJoinExec> {
        SortMergeJoinExec::try_new(left, right, on, join_type)
    }

    async fn join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
    ) -> Result<(Vec<String>, Vec<RecordBatch>)> {
        let join = join(left, right, on, join_type)?;
        let columns = columns(&join.schema());

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        Ok((columns, batches))
    }

    async fn partitioned_join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
    ) -> Result<(Vec<String>, Vec<RecordBatch>)> {
        let partition_count = 4;

        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| {
                (
                    Arc::new(l.clone()) as Arc<dyn PhysicalExpr>,
                    Arc::new(r.clone()) as Arc<dyn PhysicalExpr>,
                )
            })
            .unzip();

        let join = SortMergeJoinExec::try_new(
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::Hash(left_expr, partition_count),
            )?),
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::Hash(right_expr, partition_count),
            )?),
            on,
            join_type,
        )?;

        let columns = columns(&join.schema());

        let mut batches = vec![];
        for i in 0..partition_count {
            let stream = join.execute(i).await?;
            let more_batches = common::collect(stream).await?;
            batches.extend(
                more_batches
                    .into_iter()
                    .filter(|b| b.num_rows() > 0)
                    .collect::<Vec<_>>(),
            );
        }

        Ok((columns, batches))
    }

    #[tokio::test]
    async fn join_inner_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );

        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) =
            join_collect(left.clone(), right.clone(), on.clone(), &JoinType::Inner)
                .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_inner_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = partitioned_join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Inner,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_one_no_shared_column_names() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b2", &right.schema())?,
        )];

        let (columns, batches) = join_collect(left, right, on, &JoinType::Inner).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_two() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 2]),
            ("b2", &vec![1, 2, 2]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Column::new_with_schema("a1", &left.schema())?,
                Column::new_with_schema("a1", &right.schema())?,
            ),
            (
                Column::new_with_schema("b2", &left.schema())?,
                Column::new_with_schema("b2", &right.schema())?,
            ),
        ];

        let (columns, batches) = join_collect(left, right, on, &JoinType::Inner).await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b2 | c1 | a1 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 1  | 7  | 1  | 1  | 70 |",
            "| 2  | 2  | 8  | 2  | 2  | 80 |",
            "| 2  | 2  | 9  | 2  | 2  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    /// Test where the left has 2 parts, the right with 1 part => 1 part
    #[tokio::test]
    async fn join_inner_one_two_parts_left() -> Result<()> {
        let batch1 = build_table_i32(
            ("a1", &vec![1, 2]),
            ("b2", &vec![1, 2]),
            ("c1", &vec![7, 8]),
        );
        let batch2 =
            build_table_i32(("a1", &vec![2]), ("b2", &vec![2]), ("c1", &vec![9]));
        let schema = batch1.schema().clone();
        let left = Arc::new(
            MemoryExec::try_new(&[vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Column::new_with_schema("a1", &left.schema())?,
                Column::new_with_schema("a1", &right.schema())?,
            ),
            (
                Column::new_with_schema("b2", &left.schema())?,
                Column::new_with_schema("b2", &right.schema())?,
            ),
        ];

        let (columns, batches) = join_collect(left, right, on, &JoinType::Inner).await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b2 | c1 | a1 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 1  | 7  | 1  | 1  | 70 |",
            "| 2  | 2  | 8  | 2  | 2  | 80 |",
            "| 2  | 2  | 9  | 2  | 2  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    /// Test where the left has 1 part, the right has 2 parts => 2 parts
    #[tokio::test]
    async fn join_inner_one_two_parts_right() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );

        let batch1 = build_table_i32(
            ("a2", &vec![10, 20]),
            ("b1", &vec![4, 6]),
            ("c2", &vec![70, 80]),
        );
        let batch2 =
            build_table_i32(("a2", &vec![30]), ("b1", &vec![5]), ("c2", &vec![90]));
        let schema = batch1.schema().clone();
        let right = Arc::new(
            MemoryExec::try_new(&[vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Inner)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        // first part
        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        // second part
        let stream = join.execute(1).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 1);
        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 2  | 5  | 8  | 30 | 5  | 90 |",
            "| 3  | 5  | 9  | 30 | 5  | 90 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    fn build_table_two_batches(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema().clone();
        Arc::new(
            MemoryExec::try_new(&[vec![batch.clone(), batch]], schema, None).unwrap(),
        )
    }

    #[tokio::test]
    async fn join_left_multi_batch() {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b1", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Left).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let stream = join.execute(0).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    | 7  |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_full_multi_batch() {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        // create two identical batches for the right side
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Full).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_left_empty_right() {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b1", right.schema()).unwrap(),
        )];
        let schema = right.schema().clone();
        let right = Arc::new(MemoryExec::try_new(&[vec![right]], schema, None).unwrap());
        let join = join(left, right, on, &JoinType::Left).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let stream = join.execute(0).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  |    | 4  |    |",
            "| 2  | 5  | 8  |    | 5  |    |",
            "| 3  | 7  | 9  |    | 7  |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_full_empty_right() {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b2", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", right.schema()).unwrap(),
        )];
        let schema = right.schema().clone();
        let right = Arc::new(MemoryExec::try_new(&[vec![right]], schema, None).unwrap());
        let join = join(left, right, on, &JoinType::Full).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  |    |    |    |",
            "| 2  | 5  | 8  |    |    |    |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_left_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) =
            join_collect(left.clone(), right.clone(), on.clone(), &JoinType::Left)
                .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    | 7  |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_left_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = partitioned_join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Left,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    | 7  |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_semi() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 2, 3]),
            ("b1", &vec![4, 5, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 5, 6, 5]), // 5 is double on the right
            ("c2", &vec![70, 80, 90, 100]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Semi)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 1  | 4  | 7  |",
            "| 2  | 5  | 8  |",
            "| 2  | 5  | 8  |",
            "+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_anti() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 2, 3, 5]),
            ("b1", &vec![4, 5, 5, 7, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 8, 9, 11]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 5, 6, 5]), // 5 is double on the right
            ("c2", &vec![70, 80, 90, 100]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Anti)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 3  | 7  | 9  |",
            "| 5  | 7  | 11 |",
            "+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn join_right_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = join_collect(left, right, on, &JoinType::Right).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "|    | 6  |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_right_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) =
            partitioned_join_collect(left, right, on, &JoinType::Right).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "|    | 6  |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_full_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Full)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }
}
