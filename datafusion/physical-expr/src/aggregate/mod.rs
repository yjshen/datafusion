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

use crate::PhysicalExpr;
use arrow::datatypes::Field;
use datafusion_common::Result;
use datafusion_expr::Accumulator;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

mod approx_distinct;
mod approx_median;
mod approx_percentile_cont;
mod approx_percentile_cont_with_weight;
mod array_agg;
mod average;
mod coercion_rule;
mod correlation;
mod count;
mod covariance;
mod distinct_expressions;
#[macro_use]
mod min_max;
pub mod build_in;
mod hyperloglog;
mod stats;
mod stddev;
mod sum;
mod tdigest;
mod variance;

/// Module with some convenient methods used in expression building
pub mod helpers {
    pub use min_max::{max, min};
}

pub use approx_distinct::ApproxDistinct;
pub use approx_median::ApproxMedian;
pub use approx_percentile_cont::ApproxPercentileCont;
pub use approx_percentile_cont_with_weight::ApproxPercentileContWithWeight;
pub use array_agg::ArrayAgg;
pub use average::{Avg, AvgAccumulator};
pub use correlation::Correlation;
pub use count::Count;
pub use covariance::{Covariance, CovariancePop};
pub use distinct_expressions::{DistinctArrayAgg, DistinctCount};
pub use min_max::{Max, Min};
pub use min_max::{MaxAccumulator, MinAccumulator};
pub use stats::StatsType;
pub use stddev::{Stddev, StddevPop};
pub use sum::Sum;
pub use variance::{Variance, VariancePop};

/// An aggregate expression that:
/// * knows its resulting field
/// * knows how to create its accumulator
/// * knows its accumulator's state's field
/// * knows the expressions from whose its accumulator will receive values
pub trait AggregateExpr: Send + Sync + Debug {
    /// Returns the aggregate expression as [`Any`](std::any::Any) so that it can be
    /// downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    /// the field of the final result of this aggregation.
    fn field(&self) -> Result<Field>;

    /// the accumulator used to accumulate values from the expressions.
    /// the accumulator expects the same number of arguments as `expressions` and must
    /// return states with the same description as `state_fields`
    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>>;

    /// the fields that encapsulate the Accumulator's state
    /// the number of fields here equals the number of states that the accumulator contains
    fn state_fields(&self) -> Result<Vec<Field>>;

    /// expressions that are passed to the Accumulator.
    /// Single-column aggregations such as `sum` return a single value, others (e.g. `cov`) return many.
    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>>;

    /// Human readable name such as `"MIN(c2)"`. The default
    /// implementation returns placeholder text.
    fn name(&self) -> &str {
        "AggregateExpr: default name"
    }
}
