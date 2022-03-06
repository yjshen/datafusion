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

//! Declaration of built-in (aggregate) functions.
//! This module contains built-in aggregates' enumeration and metadata.
//!
//! Generally, an aggregate has:
//! * a signature
//! * a return type, that is a function of the incoming argument's types
//! * the computation, that must accept each valid signature
//!
//! * Signature: see `Signature`
//! * Return type: a function `(arg_types) -> return_type`. E.g. for min, ([f32]) -> f32, ([f64]) -> f64.

use crate::coercion_rule::aggregate_rule::coerce_types;
use crate::expressions::{
    avg_return_type, correlation_return_type, covariance_return_type, stddev_return_type,
    sum_return_type, variance_return_type,
};
use arrow::datatypes::{DataType, Field, TimeUnit};
use datafusion_common::Result;
use datafusion_expr::{AggregateFunction, Signature, TypeSignature, Volatility};

/// Returns the datatype of the aggregate function.
/// This is used to get the returned data type for aggregate expr.
pub fn return_type(
    fun: &AggregateFunction,
    input_expr_types: &[DataType],
) -> Result<DataType> {
    // Note that this function *must* return the same type that the respective physical expression returns
    // or the execution panics.

    let coerced_data_types = coerce_types(fun, input_expr_types, &signature(fun))?;

    match fun {
        // TODO If the datafusion is compatible with PostgreSQL, the returned data type should be INT64.
        AggregateFunction::Count | AggregateFunction::ApproxDistinct => {
            Ok(DataType::UInt64)
        }
        AggregateFunction::Max | AggregateFunction::Min => {
            // For min and max agg function, the returned type is same as input type.
            // The coerced_data_types is same with input_types.
            Ok(coerced_data_types[0].clone())
        }
        AggregateFunction::Sum => sum_return_type(&coerced_data_types[0]),
        AggregateFunction::Variance => variance_return_type(&coerced_data_types[0]),
        AggregateFunction::VariancePop => variance_return_type(&coerced_data_types[0]),
        AggregateFunction::Covariance => covariance_return_type(&coerced_data_types[0]),
        AggregateFunction::CovariancePop => {
            covariance_return_type(&coerced_data_types[0])
        }
        AggregateFunction::Correlation => correlation_return_type(&coerced_data_types[0]),
        AggregateFunction::Stddev => stddev_return_type(&coerced_data_types[0]),
        AggregateFunction::StddevPop => stddev_return_type(&coerced_data_types[0]),
        AggregateFunction::Avg => avg_return_type(&coerced_data_types[0]),
        AggregateFunction::ArrayAgg => Ok(DataType::List(Box::new(Field::new(
            "item",
            coerced_data_types[0].clone(),
            true,
        )))),
        AggregateFunction::ApproxPercentileCont => Ok(coerced_data_types[0].clone()),
        AggregateFunction::ApproxMedian => Ok(coerced_data_types[0].clone()),
    }
}

pub static STRINGS: &[DataType] = &[DataType::Utf8, DataType::LargeUtf8];

pub static NUMERICS: &[DataType] = &[
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float32,
    DataType::Float64,
];

pub static TIMESTAMPS: &[DataType] = &[
    DataType::Timestamp(TimeUnit::Second, None),
    DataType::Timestamp(TimeUnit::Millisecond, None),
    DataType::Timestamp(TimeUnit::Microsecond, None),
    DataType::Timestamp(TimeUnit::Nanosecond, None),
];

pub static DATES: &[DataType] = &[DataType::Date32, DataType::Date64];

/// the signatures supported by the function `fun`.
pub fn signature(fun: &AggregateFunction) -> Signature {
    // note: the physical expression must accept the type returned by this function or the execution panics.
    match fun {
        AggregateFunction::Count
        | AggregateFunction::ApproxDistinct
        | AggregateFunction::ArrayAgg => Signature::any(1, Volatility::Immutable),
        AggregateFunction::Min | AggregateFunction::Max => {
            let valid = STRINGS
                .iter()
                .chain(NUMERICS.iter())
                .chain(TIMESTAMPS.iter())
                .chain(DATES.iter())
                .cloned()
                .collect::<Vec<_>>();
            Signature::uniform(1, valid, Volatility::Immutable)
        }
        AggregateFunction::Avg
        | AggregateFunction::Sum
        | AggregateFunction::Variance
        | AggregateFunction::VariancePop
        | AggregateFunction::Stddev
        | AggregateFunction::StddevPop
        | AggregateFunction::ApproxMedian => {
            Signature::uniform(1, NUMERICS.to_vec(), Volatility::Immutable)
        }
        AggregateFunction::Covariance | AggregateFunction::CovariancePop => {
            Signature::uniform(2, NUMERICS.to_vec(), Volatility::Immutable)
        }
        AggregateFunction::Correlation => {
            Signature::uniform(2, NUMERICS.to_vec(), Volatility::Immutable)
        }
        AggregateFunction::ApproxPercentileCont => Signature::one_of(
            // Accept any numeric value paired with a float64 percentile
            NUMERICS
                .iter()
                .map(|t| TypeSignature::Exact(vec![t.clone(), DataType::Float64]))
                .collect(),
            Volatility::Immutable,
        ),
    }
}
