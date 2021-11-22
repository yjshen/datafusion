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

use std::iter::repeat;
use std::sync::Arc;
use std::vec;
use std::{any::Any, usize};

use arrow::array::*;
use arrow::datatypes::*;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::StreamExt;

use crate::arrow_dyn_list_array::DynMutableListArray;
use crate::error::{DataFusionError, Result};
use crate::execution::runtime_env::RuntimeEnv;
use crate::execution::runtime_env::RUNTIME_ENV;
use crate::logical_plan::JoinType;
use crate::physical_plan::expressions::{exprs_to_sort_columns, PhysicalSortExpr};
use crate::physical_plan::joins::{
    build_join_schema, check_join_is_valid, column_indices_from_schema, comp_rows,
    equal_rows, ColumnIndex, JoinOn,
};
use crate::physical_plan::sorts::external_sort::ExternalSortExec;
use crate::physical_plan::stream::RecordBatchReceiverStream;
use crate::physical_plan::Statistics;
use crate::physical_plan::{
    expressions::Column,
    metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
};
use crate::physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
};
use arrow::compute::partition::lexicographical_partition_ranges;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::Range;
use tokio::sync::mpsc::{Receiver, Sender};

fn join_arrays(rb: &RecordBatch, on_column: &Vec<Column>) -> Vec<ArrayRef> {
    on_column
        .iter()
        .map(|c| rb.column(c.index()).clone())
        .collect()
}

fn range_start_indices(buffered_ranges: &VecDeque<Range<usize>>) -> Vec<usize> {
    let mut idx = 0;
    let mut start_indices: Vec<usize> = vec![];
    buffered_ranges.iter().for_each(|r| {
        start_indices.push(idx);
        idx += r.len();
    });
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
        .enumerate()
        .find(|(_, start_idx)| **start_idx >= idx)
        .unwrap();
    let mut batch_idx = if *find.1 == idx { find.0 } else { find.0 - 1 };

    while remaining > 0 {
        let current_range = &buffered_ranges[batch_idx];
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

/// Slice of batch at `batch_idx` inside BufferedBatches.
struct Slice {
    batch_idx: usize,
    start_idx: usize,
    len: usize,
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
    ) -> Result<Option<Self>> {
        match batch {
            Some(batch) => {
                let columns = exprs_to_sort_columns(&batch, expr)?;
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
    fn is_last_range(&self, range: &Range<usize>) -> bool {
        range.end == self.batch.num_rows()
    }
}

struct StreamingBatch {
    batch: Option<PartitionedRecordBatch>,
    cur_row: usize,
    cur_range: usize,
    num_rows: usize,
    num_ranges: usize,
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
                for c in &self.on_column {
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
            let ranges = &self.batch.as_ref().unwrap().ranges;
            if self.cur_row == ranges[self.cur_range + 1].start {
                self.cur_range += 1;
                self.is_new_key = true;
            }
        } else {
            self.batch = None;
        }
    }

    fn advance_key(&mut self) {
        let ranges = &self.batch.as_ref().unwrap().ranges;
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
    next_key_batch: Vec<PartitionedRecordBatch>,
    /// Join on column
    on_column: Vec<Column>,
    sort: Vec<PhysicalSortExpr>,
}

impl BufferedBatches {
    fn new(on_column: Vec<Column>, sort: Vec<PhysicalSortExpr>) -> Self {
        Self {
            batches: VecDeque::new(),
            ranges: VecDeque::new(),
            key_idx: None,
            row_num: 0,
            next_key_batch: vec![],
            on_column,
            sort,
        }
    }

    fn key_any_null(&self) -> bool {
        match &self.key_idx {
            None => return true,
            Some(key_idx) => {
                let first_batch = &self.batches[0].batch;
                for c in &self.on_column {
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
                let equal = equal_rows(key_idx, 0, &key_arrays, &current_arrays)?;
                if equal {
                    self.batches.push_back(prb.clone());
                    self.ranges.push_back(first_range.clone());
                    self.row_num += range_len;
                    Ok(single_range)
                } else {
                    self.next_key_batch.push(prb.clone());
                    Ok(false) // running key ends
                }
            }
        }
    }

    fn cleanup(&mut self) {
        self.batches.drain(..);
        self.ranges.drain(..);
        self.next_key_batch.drain(..);
    }

    fn reset_batch(&mut self, prb: &PartitionedRecordBatch) {
        self.cleanup();
        self.batches.push_back(prb.clone());
        let first_range = &prb.ranges[0];
        self.ranges.push_back(first_range.clone());
        self.key_idx = Some(0);
        self.row_num = first_range.len();
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
            let next_range_idx = batch
                .ranges
                .iter()
                .enumerate()
                .find(|(_, range)| range.start == tail_range.start)
                .unwrap()
                .0;
            self.key_idx = Some(tail_range.end);
            self.ranges.push_back(batch.ranges[next_range_idx].clone());
            self.row_num = batch.ranges[next_range_idx].len();
            self.batches.push_back(batch);
        }
    }
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
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "making mutable of type {} is not implemented yet",
                    data_type
                )));
            }
        },
    })
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

struct OutputBuffer {
    arrays: Vec<Box<dyn MutableArray>>,
    target_batch_size: usize,
    slots_available: usize,
    schema: Arc<Schema>,
}

impl OutputBuffer {
    fn new(target_batch_size: usize, schema: Arc<Schema>) -> Result<Self> {
        let arrays = new_arrays(&schema, target_batch_size)
            .map_err(DataFusionError::ArrowError)?;
        Ok(Self {
            arrays,
            target_batch_size,
            slots_available: target_batch_size,
            schema,
        })
    }

    fn output_and_reset(&mut self) -> ArrowResult<RecordBatch> {
        let result = make_batch(self.schema.clone(), self.arrays.drain(..).collect());
        let mut new = new_arrays(&self.schema, self.target_batch_size)?;
        self.arrays.append(&mut new);
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

macro_rules! repeat_n {
    ($TO:ty, $FROM:ty, $N:expr, $to: ident, $from: ident, $idx: ident) => {{
        let to = $to.as_mut_any().downcast_mut::<$TO>().unwrap();
        let from = $from
            .as_any()
            .downcast_ref::<$FROM>()
            .unwrap()
            .slice($idx, 1);
        let repeat_iter = from
            .iter()
            .flat_map(|v| repeat(v).take($N))
            .collect::<Vec<_>>();
        to.extend_trusted_len(repeat_iter.into_iter());
    }};
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

struct SortMergeJoinDriver {
    streamed: SendableRecordBatchStream,
    buffered: SendableRecordBatchStream,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    stream_batch: StreamingBatch,
    buffered_batches: BufferedBatches,
    output: OutputBuffer,
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
    ) -> Result<Self> {
        let batch_size = runtime.batch_size();
        Ok(Self {
            streamed,
            buffered,
            column_indices,
            stream_batch: StreamingBatch::new(on_streamed, streamed_sort),
            buffered_batches: BufferedBatches::new(on_buffered, buffered_sort),
            output: OutputBuffer::new(batch_size, schema)?,
        })
    }

    async fn inner_join_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        while self.find_inner_next().await? {
            loop {
                self.join_eq_records(sender).await?;
                self.stream_batch.advance();
                if self.stream_batch.is_new_key {
                    break;
                }
            }
        }

        Ok(())
    }

    async fn outer_join_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let mut buffer_ends = false;

        loop {
            let OuterMatchResult {
                get_match,
                buffered_ended,
                more_output,
            } = self.find_outer_next(buffer_ends).await?;
            if !more_output {
                break;
            }
            buffer_ends = buffered_ended;
            if get_match {
                loop {
                    self.join_eq_records(sender).await?;
                    self.stream_batch.advance();
                    if self.stream_batch.is_new_key {
                        break;
                    }
                }
            } else {
                self.stream_copy_buffer_null(sender).await?;
            }
        }

        Ok(())
    }

    async fn full_outer_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let mut stream_ends = false;
        let mut buffer_ends = false;
        let mut advance_stream = true;
        let mut advance_buffer = true;

        loop {
            if advance_buffer {
                buffer_ends = !self.advance_buffered_key().await?;
            }
            if advance_stream {
                stream_ends = !self.advance_streamed_key().await?;
            }

            if stream_ends && buffer_ends {
                break;
            } else if stream_ends {
                self.stream_null_buffer_copy(sender).await?;
                advance_buffer = true;
                advance_stream = false;
            } else if buffer_ends {
                self.stream_copy_buffer_null(sender).await?;
                advance_stream = true;
                advance_buffer = false;
            } else {
                if self.stream_batch.key_any_null() {
                    self.stream_copy_buffer_null(sender).await?;
                    advance_stream = true;
                    advance_buffer = false;
                    continue;
                }
                if self.buffered_batches.key_any_null() {
                    self.stream_null_buffer_copy(sender).await?;
                    advance_buffer = true;
                    advance_stream = false;
                    continue;
                }

                let current_cmp = self.compare_stream_buffer()?;
                match current_cmp {
                    Ordering::Less => {
                        self.stream_copy_buffer_null(sender).await?;
                        advance_stream = true;
                        advance_buffer = false;
                    }
                    Ordering::Equal => {
                        loop {
                            self.join_eq_records(sender).await?;
                            self.stream_batch.advance();
                            if self.stream_batch.is_new_key {
                                break;
                            }
                        }
                        advance_stream = false; // we already reach the next key of stream
                        advance_buffer = true;
                    }
                    Ordering::Greater => {
                        self.stream_null_buffer_copy(sender).await?;
                        advance_buffer = true;
                        advance_stream = false;
                    }
                }
            }
        }
        Ok(())
    }

    async fn semi_join_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        while self.find_inner_next().await? {
            self.stream_copy_buffer_omit(sender).await?;
        }
        Ok(())
    }

    async fn anti_join_driver(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let mut buffer_ends = false;

        loop {
            let OuterMatchResult {
                get_match,
                buffered_ended,
                more_output,
            } = self.find_outer_next(buffer_ends).await?;
            if !more_output {
                break;
            }
            buffer_ends = buffered_ended;
            if get_match {
                // do nothing
            } else {
                self.stream_copy_buffer_omit(sender).await?;
            }
        }

        Ok(())
    }

    async fn join_eq_records(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let mut remaining = self.buffered_batches.row_num;
        let stream_batch = &self.stream_batch.batch.as_ref().unwrap().batch;
        let stream_row = self.stream_batch.cur_row;

        let batches = self
            .buffered_batches
            .batches
            .iter()
            .map(|prb| &prb.batch)
            .collect::<Vec<_>>();
        let buffered_ranges = &self.buffered_batches.ranges;

        let mut unfinished = true;
        let mut buffered_idx = 0;
        let start_indices = range_start_indices(buffered_ranges);

        // output each buffered matching record once
        while unfinished {
            let output_slots_available = self.output.slots_available;
            let rows_to_output = if output_slots_available >= remaining {
                unfinished = false;
                remaining
            } else {
                remaining -= output_slots_available;
                output_slots_available
            };

            // get slices for buffered side for the current output
            let slices = slices_from_batches(
                buffered_ranges,
                &start_indices,
                buffered_idx,
                rows_to_output,
            );

            self.output
                .arrays
                .iter_mut()
                .zip(self.column_indices.iter())
                .for_each(|(array, column_index)| {
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
                        copy_slices(&batches, &slices, array, column_index);
                    }
                });

            self.output.append(rows_to_output);
            buffered_idx += rows_to_output;

            if self.output.is_full() {
                let result = self.output.output_and_reset();
                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via inner join stream: {}", e);
                };
            }
        }
        Ok(())
    }

    async fn stream_copy_buffer_null(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let stream_batch = &self.stream_batch.batch.as_ref().unwrap().batch;
        let batch = vec![stream_batch];
        let stream_range = &self.stream_batch.batch.as_ref().unwrap().ranges
            [self.stream_batch.cur_range];
        let mut remaining = stream_range.len();

        let mut unfinished = true;
        let mut streamed_idx = self.stream_batch.cur_row;

        // output each buffered matching record once
        while unfinished {
            let output_slots_available = self.output.slots_available;
            let rows_to_output = if output_slots_available >= remaining {
                unfinished = false;
                remaining
            } else {
                remaining -= output_slots_available;
                output_slots_available
            };

            let slice = vec![Slice {
                batch_idx: 0,
                start_idx: streamed_idx,
                len: rows_to_output,
            }];

            self.output
                .arrays
                .iter_mut()
                .zip(self.column_indices.iter())
                .for_each(|(array, column_index)| {
                    if column_index.is_left {
                        copy_slices(&batch, &slice, array, column_index);
                    } else {
                        (0..rows_to_output).for_each(|_| array.push_null());
                    }
                });

            self.output.append(rows_to_output);
            streamed_idx += rows_to_output;

            if self.output.is_full() {
                let result = self.output.output_and_reset();
                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via outer join stream: {}", e);
                };
            }
        }
        Ok(())
    }

    async fn stream_null_buffer_copy(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let mut remaining = self.buffered_batches.row_num;

        let batches = self
            .buffered_batches
            .batches
            .iter()
            .map(|prb| &prb.batch)
            .collect::<Vec<_>>();
        let buffered_ranges = &self.buffered_batches.ranges;

        let mut unfinished = true;
        let mut buffered_idx = 0;
        let start_indices = range_start_indices(buffered_ranges);

        // output each buffered matching record once
        while unfinished {
            let output_slots_available = self.output.slots_available;
            let rows_to_output = if output_slots_available >= remaining {
                unfinished = false;
                remaining
            } else {
                remaining -= output_slots_available;
                output_slots_available
            };

            // get slices for buffered side for the current output
            let slices = slices_from_batches(
                buffered_ranges,
                &start_indices,
                buffered_idx,
                rows_to_output,
            );

            self.output
                .arrays
                .iter_mut()
                .zip(self.column_indices.iter())
                .for_each(|(array, column_index)| {
                    if column_index.is_left {
                        (0..rows_to_output).for_each(|_| array.push_null());
                    } else {
                        // copy buffered start from: `buffered_idx`, len: `rows_to_output`
                        copy_slices(&batches, &slices, array, column_index);
                    }
                });

            self.output.append(rows_to_output);
            buffered_idx += rows_to_output;

            if self.output.is_full() {
                let result = self.output.output_and_reset();
                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via outer join stream: {}", e);
                };
            }
        }
        Ok(())
    }

    async fn stream_copy_buffer_omit(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        let stream_batch = &self.stream_batch.batch.as_ref().unwrap().batch;
        let batch = vec![stream_batch];
        let stream_range = &self.stream_batch.batch.as_ref().unwrap().ranges
            [self.stream_batch.cur_range];
        let mut remaining = stream_range.len();

        let mut unfinished = true;
        let mut streamed_idx = self.stream_batch.cur_row;

        // output each buffered matching record once
        while unfinished {
            let output_slots_available = self.output.slots_available;
            let rows_to_output = if output_slots_available >= remaining {
                unfinished = false;
                remaining
            } else {
                remaining -= output_slots_available;
                output_slots_available
            };

            let slice = vec![Slice {
                batch_idx: 0,
                start_idx: streamed_idx,
                len: rows_to_output,
            }];

            self.output
                .arrays
                .iter_mut()
                .zip(self.column_indices.iter())
                .for_each(|(array, column_index)| {
                    copy_slices(&batch, &slice, array, column_index);
                });

            self.output.append(rows_to_output);
            streamed_idx += rows_to_output;

            if self.output.is_full() {
                let result = self.output.output_and_reset();
                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via semi/anti join stream: {}", e);
                };
            }
        }
        Ok(())
    }

    async fn find_inner_next(&mut self) -> Result<bool> {
        if self.stream_batch.key_any_null() {
            let more_stream = self.advance_streamed_key_null_free().await?;
            if !more_stream {
                return Ok(false);
            }
        }

        if self.buffered_batches.key_any_null() {
            let more_buffer = self.advance_buffered_key_null_free().await?;
            if !more_buffer {
                return Ok(false);
            }
        }

        loop {
            let current_cmp = self.compare_stream_buffer()?;
            match current_cmp {
                Ordering::Less => {
                    let more_stream = self.advance_streamed_key_null_free().await?;
                    if !more_stream {
                        return Ok(false);
                    }
                }
                Ordering::Equal => return Ok(true),
                Ordering::Greater => {
                    let more_buffer = self.advance_buffered_key_null_free().await?;
                    if !more_buffer {
                        return Ok(false);
                    }
                }
            }
        }
    }

    async fn find_outer_next(&mut self, buffer_ends: bool) -> Result<OuterMatchResult> {
        let more_stream = self.advance_streamed_key().await?;
        if buffer_ends {
            return Ok(OuterMatchResult {
                get_match: false,
                buffered_ended: true,
                more_output: more_stream,
            });
        } else {
            if !more_stream {
                return Ok(OuterMatchResult {
                    get_match: false,
                    buffered_ended: false,
                    more_output: false,
                });
            }

            if self.buffered_batches.key_any_null() {
                let more_buffer = self.advance_buffered_key_null_free().await?;
                if !more_buffer {
                    return Ok(OuterMatchResult {
                        get_match: false,
                        buffered_ended: true,
                        more_output: true,
                    });
                }
            }

            loop {
                if self.stream_batch.key_any_null() {
                    return Ok(OuterMatchResult {
                        get_match: false,
                        buffered_ended: false,
                        more_output: true,
                    });
                }

                let current_cmp = self.compare_stream_buffer()?;
                match current_cmp {
                    Ordering::Less => {
                        return Ok(OuterMatchResult {
                            get_match: false,
                            buffered_ended: false,
                            more_output: true,
                        })
                    }
                    Ordering::Equal => {
                        return Ok(OuterMatchResult {
                            get_match: true,
                            buffered_ended: false,
                            more_output: true,
                        })
                    }
                    Ordering::Greater => {
                        let more_buffer = self.advance_buffered_key_null_free().await?;
                        if !more_buffer {
                            return Ok(OuterMatchResult {
                                get_match: false,
                                buffered_ended: true,
                                more_output: true,
                            });
                        }
                    }
                }
            }
        }
    }

    async fn get_stream_next(&mut self) -> Result<()> {
        let batch = self.streamed.next().await.transpose()?;
        let prb = PartitionedRecordBatch::new(batch, &self.stream_batch.sort)?;
        self.stream_batch.rest_batch(prb);
        Ok(())
    }

    /// true for has next, false for ended
    async fn advance_streamed_key(&mut self) -> Result<bool> {
        if self.stream_batch.is_finished() || self.stream_batch.is_last_key_in_batch() {
            self.get_stream_next().await?;
            Ok(!self.stream_batch.is_finished())
        } else {
            self.stream_batch.advance_key();
            Ok(true)
        }
    }

    /// true for has next, false for ended
    async fn advance_streamed_key_null_free(&mut self) -> Result<bool> {
        let mut more_stream_keys = self.advance_streamed_key().await?;
        loop {
            if more_stream_keys && self.stream_batch.key_any_null() {
                more_stream_keys = self.advance_streamed_key().await?;
            } else {
                break;
            }
        }
        Ok(more_stream_keys)
    }

    async fn get_buffered_next(&mut self) -> Result<Option<PartitionedRecordBatch>> {
        let batch = self.buffered.next().await.transpose()?;
        PartitionedRecordBatch::new(batch, &self.buffered_batches.sort)
    }

    /// true for has next, false for ended
    async fn advance_buffered_key(&mut self) -> Result<bool> {
        if self.buffered_batches.is_finished()? {
            if self.buffered_batches.next_key_batch.is_empty() {
                let batch = self.get_buffered_next().await?;
                match batch {
                    None => return Ok(false),
                    Some(batch) => {
                        self.buffered_batches.reset_batch(&batch);
                        if batch.ranges.len() == 1 {
                            self.cumulate_same_keys().await?;
                        }
                    }
                }
            } else {
                assert_eq!(self.buffered_batches.next_key_batch.len(), 1);
                let batch = self.buffered_batches.next_key_batch.pop().unwrap();
                self.buffered_batches.reset_batch(&batch);
                if batch.ranges.len() == 1 {
                    self.cumulate_same_keys().await?;
                }
            }
        } else {
            self.buffered_batches.advance_in_current_batch();
            if self.buffered_batches.batches[0]
                .is_last_range(&self.buffered_batches.ranges[0])
            {
                self.cumulate_same_keys().await?;
            }
        }
        Ok(false)
    }

    /// true for has next, false for buffer side ended
    async fn cumulate_same_keys(&mut self) -> Result<bool> {
        loop {
            let batch = self.get_buffered_next().await?;
            match batch {
                None => return Ok(false),
                Some(batch) => {
                    let more_batches = self.buffered_batches.running_key(&batch)?;
                    if !more_batches {
                        return Ok(true);
                    }
                }
            }
        }
    }

    async fn advance_buffered_key_null_free(&mut self) -> Result<bool> {
        let mut more_buffered_keys = self.advance_buffered_key().await?;
        loop {
            if more_buffered_keys && self.buffered_batches.key_any_null() {
                more_buffered_keys = self.advance_buffered_key().await?;
            } else {
                break;
            }
        }
        Ok(more_buffered_keys)
    }

    fn compare_stream_buffer(&self) -> Result<Ordering> {
        let stream_arrays = join_arrays(
            &self.stream_batch.batch.as_ref().unwrap().batch,
            &self.stream_batch.on_column,
        );
        let buffer_arrays = join_arrays(
            &self.buffered_batches.batches[0].batch,
            &self.buffered_batches.on_column,
        );
        comp_rows(
            self.stream_batch.cur_row,
            self.buffered_batches.key_idx.unwrap(),
            &stream_arrays,
            &buffer_arrays,
        )
    }
}

struct OuterMatchResult {
    get_match: bool,
    buffered_ended: bool,
    more_output: bool,
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
    #[allow(dead_code)]
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

    fn output_partitioning(&self) -> Partitioning {
        self.right.output_partitioning()
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
            )?,
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
            )?,
        };

        match self.join_type {
            JoinType::Inner => driver.inner_join_driver(&tx).await?,
            JoinType::Left => driver.outer_join_driver(&tx).await?,
            JoinType::Right => driver.outer_join_driver(&tx).await?,
            JoinType::Full => driver.full_outer_driver(&tx).await?,
            JoinType::Semi => driver.semi_join_driver(&tx).await?,
            JoinType::Anti => driver.anti_join_driver(&tx).await?,
        }

        let result = RecordBatchReceiverStream::create(&self.schema, rx);

        Ok(result)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
    use crate::physical_plan::PhysicalExpr;

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
