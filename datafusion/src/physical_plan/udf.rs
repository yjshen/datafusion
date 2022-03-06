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

//! UDF support

use crate::error::Result;
use crate::physical_plan::functions::ScalarFunctionExpr;
use crate::physical_plan::PhysicalExpr;
use arrow::datatypes::Schema;
use datafusion_physical_expr::coercion_rule::type_coercion::coerce;

pub use datafusion_expr::ScalarUDF;

use std::sync::Arc;

/// Create a physical expression of the UDF.
/// This function errors when `args`' can't be coerced to a valid argument type of the UDF.
pub fn create_physical_expr(
    fun: &ScalarUDF,
    input_phy_exprs: &[Arc<dyn PhysicalExpr>],
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    // coerce
    let coerced_phy_exprs = coerce(input_phy_exprs, input_schema, &fun.signature)?;

    let coerced_exprs_types = coerced_phy_exprs
        .iter()
        .map(|e| e.data_type(input_schema))
        .collect::<Result<Vec<_>>>()?;

    Ok(Arc::new(ScalarFunctionExpr::new(
        &fun.name,
        fun.fun.clone(),
        coerced_phy_exprs,
        (fun.return_type)(&coerced_exprs_types)?.as_ref(),
    )))
}
