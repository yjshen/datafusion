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

//! [`ScalarUDFImpl`] definitions for range and gen_series functions.

use crate::utils::make_scalar_function;
use arrow::array::{Array, ArrayRef, Date32Builder, Int64Array, ListArray, ListBuilder};
use arrow::datatypes::{DataType, Field};
use arrow_array::types::{Date32Type, IntervalMonthDayNanoType};
use arrow_array::NullArray;
use arrow_buffer::{BooleanBufferBuilder, NullBuffer, OffsetBuffer};
use arrow_schema::DataType::*;
use arrow_schema::IntervalUnit::MonthDayNano;
use datafusion_common::cast::{as_date32_array, as_int64_array, as_interval_mdn_array};
use datafusion_common::{exec_err, not_impl_datafusion_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use itertools::Itertools;
use std::any::Any;
use std::iter::from_fn;
use std::sync::Arc;

make_udf_expr_and_func!(
    Range,
    range,
    start stop step,
    "create a list of values in the range between start and stop",
    range_udf
);
#[derive(Debug)]
pub(super) struct Range {
    signature: Signature,
    aliases: Vec<String>,
}
impl Range {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            aliases: vec![],
        }
    }
}
impl ScalarUDFImpl for Range {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "range"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        arg_types
            .iter()
            .map(|arg_type| match arg_type {
                Null => Ok(Null),
                Int8 => Ok(Int64),
                Int16 => Ok(Int64),
                Int32 => Ok(Int64),
                Int64 => Ok(Int64),
                UInt8 => Ok(Int64),
                UInt16 => Ok(Int64),
                UInt32 => Ok(Int64),
                UInt64 => Ok(Int64),
                Timestamp(_, _) => Ok(Date32),
                Date32 => Ok(Date32),
                Date64 => Ok(Date32),
                Utf8 => Ok(Date32),
                LargeUtf8 => Ok(Date32),
                Utf8View => Ok(Date32),
                Interval(_) => Ok(Interval(MonthDayNano)),
                _ => exec_err!("Unsupported DataType"),
            })
            .try_collect()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.iter().any(|t| t.is_null()) {
            Ok(Null)
        } else {
            Ok(List(Arc::new(Field::new(
                "item",
                arg_types[0].clone(),
                true,
            ))))
        }
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        if args.iter().any(|arg| arg.data_type().is_null()) {
            return Ok(ColumnarValue::Array(Arc::new(NullArray::new(1))));
        }
        match args[0].data_type() {
            Int64 => make_scalar_function(|args| gen_range_inner(args, false))(args),
            Date32 => make_scalar_function(|args| gen_range_date(args, false))(args),
            _ => {
                exec_err!("unsupported type for range")
            }
        }
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

make_udf_expr_and_func!(
    GenSeries,
    gen_series,
    start stop step,
    "create a list of values in the range between start and stop, include upper bound",
    gen_series_udf
);
#[derive(Debug)]
pub(super) struct GenSeries {
    signature: Signature,
    aliases: Vec<String>,
}
impl GenSeries {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            aliases: vec![],
        }
    }
}
impl ScalarUDFImpl for GenSeries {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "generate_series"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, _arg_types: &[DataType]) -> Result<Vec<DataType>> {
        _arg_types
            .iter()
            .map(|arg_type| match arg_type {
                Null => Ok(Null),
                Int8 => Ok(Int64),
                Int16 => Ok(Int64),
                Int32 => Ok(Int64),
                Int64 => Ok(Int64),
                UInt8 => Ok(Int64),
                UInt16 => Ok(Int64),
                UInt32 => Ok(Int64),
                UInt64 => Ok(Int64),
                Timestamp(_, _) => Ok(Date32),
                Date32 => Ok(Date32),
                Date64 => Ok(Date32),
                Utf8 => Ok(Date32),
                LargeUtf8 => Ok(Date32),
                Utf8View => Ok(Date32),
                Interval(_) => Ok(Interval(MonthDayNano)),
                _ => exec_err!("Unsupported DataType"),
            })
            .try_collect()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.iter().any(|t| t.is_null()) {
            Ok(Null)
        } else {
            Ok(List(Arc::new(Field::new(
                "item",
                arg_types[0].clone(),
                true,
            ))))
        }
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        if args.iter().any(|arg| arg.data_type().is_null()) {
            return Ok(ColumnarValue::Array(Arc::new(NullArray::new(1))));
        }
        match args[0].data_type() {
            Int64 => make_scalar_function(|args| gen_range_inner(args, true))(args),
            Date32 => make_scalar_function(|args| gen_range_date(args, true))(args),
            dt => {
                exec_err!(
                    "unsupported type for gen_series. Expected Int64 or Date32, got: {}",
                    dt
                )
            }
        }
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

/// Generates an array of integers from start to stop with a given step.
///
/// This function takes 1 to 3 ArrayRefs as arguments, representing start, stop, and step values.
/// It returns a `Result<ArrayRef>` representing the resulting ListArray after the operation.
///
/// # Arguments
///
/// * `args` - An array of 1 to 3 ArrayRefs representing start, stop, and step(step value can not be zero.) values.
///
/// # Examples
///
/// gen_range(3) => [0, 1, 2]
/// gen_range(1, 4) => [1, 2, 3]
/// gen_range(1, 7, 2) => [1, 3, 5]
pub(super) fn gen_range_inner(
    args: &[ArrayRef],
    include_upper: bool,
) -> Result<ArrayRef> {
    let (start_array, stop_array, step_array) = match args.len() {
        1 => (None, as_int64_array(&args[0])?, None),
        2 => (
            Some(as_int64_array(&args[0])?),
            as_int64_array(&args[1])?,
            None,
        ),
        3 => (
            Some(as_int64_array(&args[0])?),
            as_int64_array(&args[1])?,
            Some(as_int64_array(&args[2])?),
        ),
        _ => return exec_err!("gen_range expects 1 to 3 arguments"),
    };

    let mut values = vec![];
    let mut offsets = vec![0];
    let mut valid = BooleanBufferBuilder::new(stop_array.len());
    for (idx, stop) in stop_array.iter().enumerate() {
        match retrieve_range_args(start_array, stop, step_array, idx) {
            Some((_, _, 0)) => {
                return exec_err!(
                    "step can't be 0 for function {}(start [, stop, step])",
                    if include_upper {
                        "generate_series"
                    } else {
                        "range"
                    }
                );
            }
            Some((start, stop, step)) => {
                // Below, we utilize `usize` to represent steps.
                // On 32-bit targets, the absolute value of `i64` may fail to fit into `usize`.
                let step_abs = usize::try_from(step.unsigned_abs()).map_err(|_| {
                    not_impl_datafusion_err!("step {} can't fit into usize", step)
                })?;
                values.extend(
                    gen_range_iter(start, stop, step < 0, include_upper)
                        .step_by(step_abs),
                );
                offsets.push(values.len() as i32);
                valid.append(true);
            }
            // If any of the arguments is NULL, append a NULL value to the result.
            None => {
                offsets.push(values.len() as i32);
                valid.append(false);
            }
        };
    }
    let arr = Arc::new(ListArray::try_new(
        Arc::new(Field::new("item", Int64, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int64Array::from(values)),
        Some(NullBuffer::new(valid.finish())),
    )?);
    Ok(arr)
}

/// Get the (start, stop, step) args for the range and generate_series function.
/// If any of the arguments is NULL, returns None.
fn retrieve_range_args(
    start_array: Option<&Int64Array>,
    stop: Option<i64>,
    step_array: Option<&Int64Array>,
    idx: usize,
) -> Option<(i64, i64, i64)> {
    // Default start value is 0 if not provided
    let start =
        start_array.map_or(Some(0), |arr| arr.is_valid(idx).then(|| arr.value(idx)))?;
    let stop = stop?;
    // Default step value is 1 if not provided
    let step =
        step_array.map_or(Some(1), |arr| arr.is_valid(idx).then(|| arr.value(idx)))?;
    Some((start, stop, step))
}

/// Returns an iterator of i64 values from start to stop
fn gen_range_iter(
    start: i64,
    stop: i64,
    decreasing: bool,
    include_upper: bool,
) -> Box<dyn Iterator<Item = i64>> {
    match (decreasing, include_upper) {
        // Decreasing range, stop is inclusive
        (true, true) => Box::new((stop..=start).rev()),
        // Decreasing range, stop is exclusive
        (true, false) => {
            if stop == i64::MAX {
                // start is never greater than stop, and stop is exclusive,
                // so the decreasing range must be empty.
                Box::new(std::iter::empty())
            } else {
                // Increase the stop value by one to exclude it.
                // Since stop is not i64::MAX, `stop + 1` will not overflow.
                Box::new((stop + 1..=start).rev())
            }
        }
        // Increasing range, stop is inclusive
        (false, true) => Box::new(start..=stop),
        // Increasing range, stop is exclusive
        (false, false) => Box::new(start..stop),
    }
}

fn gen_range_date(args: &[ArrayRef], include_upper: bool) -> Result<ArrayRef> {
    if args.len() != 3 {
        return exec_err!("arguments length does not match");
    }
    let (start_array, stop_array, step_array) = (
        Some(as_date32_array(&args[0])?),
        as_date32_array(&args[1])?,
        Some(as_interval_mdn_array(&args[2])?),
    );

    // values are date32s
    let values_builder = Date32Builder::new();
    let mut list_builder = ListBuilder::new(values_builder);

    for (idx, stop) in stop_array.iter().enumerate() {
        let mut stop = stop.unwrap_or(0);

        let start = if let Some(start_array_values) = start_array {
            start_array_values.value(idx)
        } else {
            list_builder.append_null();
            continue;
        };

        let step = if let Some(step) = step_array {
            step.value(idx)
        } else {
            list_builder.append_null();
            continue;
        };

        let (months, days, _) = IntervalMonthDayNanoType::to_parts(step);

        if months == 0 && days == 0 {
            return exec_err!("Cannot generate date range less than 1 day.");
        }

        let neg = months < 0 || days < 0;
        if !include_upper {
            stop = Date32Type::subtract_month_day_nano(stop, step);
        }
        let mut new_date = start;

        let values = from_fn(|| {
            if (neg && new_date < stop) || (!neg && new_date > stop) {
                None
            } else {
                let current_date = new_date;
                new_date = Date32Type::add_month_day_nano(new_date, step);
                Some(Some(current_date))
            }
        });

        list_builder.append_value(values);
    }

    let arr = Arc::new(list_builder.finish());

    Ok(arr)
}
