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

pub mod cross_join;
pub mod hash_join;
pub mod sort_merge_join;

use crate::error::{DataFusionError, Result};
use crate::logical_plan::JoinType;
use crate::physical_plan::expressions::Column;
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

/// The on clause of the join, as vector of (left, right) columns.
pub type JoinOn = Vec<(Column, Column)>;
/// Reference for JoinOn.
pub type JoinOnRef<'a> = &'a [(Column, Column)];

/// Checks whether the schemas "left" and "right" and columns "on" represent a valid join.
/// They are valid whenever their columns' intersection equals the set `on`
pub fn check_join_is_valid(left: &Schema, right: &Schema, on: JoinOnRef) -> Result<()> {
    let left: HashSet<Column> = left
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();
    let right: HashSet<Column> = right
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();

    check_join_set_is_valid(&left, &right, on)
}

/// Checks whether the sets left, right and on compose a valid join.
/// They are valid whenever their intersection equals the set `on`
pub fn check_join_set_is_valid(
    left: &HashSet<Column>,
    right: &HashSet<Column>,
    on: &[(Column, Column)],
) -> Result<()> {
    let on_left = &on.iter().map(|on| on.0.clone()).collect::<HashSet<_>>();
    let left_missing = on_left.difference(left).collect::<HashSet<_>>();

    let on_right = &on.iter().map(|on| on.1.clone()).collect::<HashSet<_>>();
    let right_missing = on_right.difference(right).collect::<HashSet<_>>();

    if !left_missing.is_empty() | !right_missing.is_empty() {
        return Err(DataFusionError::Plan(format!(
                "The left or right side of the join does not have all columns on \"on\": \nMissing on the left: {:?}\nMissing on the right: {:?}",
                left_missing,
                right_missing,
            )));
    };

    let remaining = right
        .difference(on_right)
        .cloned()
        .collect::<HashSet<Column>>();

    let collisions = left.intersection(&remaining).collect::<HashSet<_>>();

    if !collisions.is_empty() {
        return Err(DataFusionError::Plan(format!(
                "The left schema and the right schema have the following columns with the same name without being on the ON statement: {:?}. Consider aliasing them.",
                collisions,
            )));
    };

    Ok(())
}

/// Creates a schema for a join operation.
/// The fields from the left side are first
pub fn build_join_schema(left: &Schema, right: &Schema, join_type: &JoinType) -> Schema {
    let fields: Vec<Field> = match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            let left_fields = left.fields().iter();
            let right_fields = right.fields().iter();
            // left then right
            left_fields.chain(right_fields).cloned().collect()
        }
        JoinType::Semi | JoinType::Anti => left.fields().clone(),
    };
    Schema::new(fields)
}

/// Calculates column indices and left/right placement on input / output schemas and jointype
pub fn column_indices_from_schema(
    join_type: &JoinType,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    schema: &Arc<Schema>,
) -> ArrowResult<Vec<ColumnIndex>> {
    let (primary_is_left, primary_schema, secondary_schema) = match join_type {
        JoinType::Inner
        | JoinType::Left
        | JoinType::Full
        | JoinType::Semi
        | JoinType::Anti => (true, left_schema, right_schema),
        JoinType::Right => (false, right_schema, left_schema),
    };
    let mut column_indices = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let (is_primary, index) = match primary_schema.index_of(field.name()) {
            Ok(i) => Ok((true, i)),
            Err(_) => {
                match secondary_schema.index_of(field.name()) {
                    Ok(i) => Ok((false, i)),
                    _ => Err(DataFusionError::Internal(
                        format!("During execution, the column {} was not found in neither the left or right side of the join", field.name()).to_string()
                    ))
                }
            }
        }.map_err(DataFusionError::into_arrow_external_error)?;

        let is_left = is_primary && primary_is_left || !is_primary && !primary_is_left;
        column_indices.push(ColumnIndex { index, is_left });
    }

    Ok(column_indices)
}

macro_rules! equal_rows_elem {
    ($array_type:ident, $l: ident, $r: ident, $left: ident, $right: ident) => {{
        let left_array = $l.as_any().downcast_ref::<$array_type>().unwrap();
        let right_array = $r.as_any().downcast_ref::<$array_type>().unwrap();

        match (left_array.is_null($left), right_array.is_null($right)) {
            (false, false) => left_array.value($left) == right_array.value($right),
            _ => false,
        }
    }};
}

/// Left and right row have equal values
fn equal_rows(
    left: usize,
    right: usize,
    left_arrays: &[ArrayRef],
    right_arrays: &[ArrayRef],
) -> Result<bool> {
    let mut err = None;
    let res = left_arrays
        .iter()
        .zip(right_arrays)
        .all(|(l, r)| match l.data_type() {
            DataType::Null => true,
            DataType::Boolean => equal_rows_elem!(BooleanArray, l, r, left, right),
            DataType::Int8 => equal_rows_elem!(Int8Array, l, r, left, right),
            DataType::Int16 => equal_rows_elem!(Int16Array, l, r, left, right),
            DataType::Int32 => equal_rows_elem!(Int32Array, l, r, left, right),
            DataType::Int64 => equal_rows_elem!(Int64Array, l, r, left, right),
            DataType::UInt8 => equal_rows_elem!(UInt8Array, l, r, left, right),
            DataType::UInt16 => equal_rows_elem!(UInt16Array, l, r, left, right),
            DataType::UInt32 => equal_rows_elem!(UInt32Array, l, r, left, right),
            DataType::UInt64 => equal_rows_elem!(UInt64Array, l, r, left, right),
            DataType::Timestamp(_, None) => {
                equal_rows_elem!(Int64Array, l, r, left, right)
            }
            DataType::Utf8 => equal_rows_elem!(StringArray, l, r, left, right),
            DataType::LargeUtf8 => equal_rows_elem!(LargeStringArray, l, r, left, right),
            _ => {
                // This is internal because we should have caught this before.
                err = Some(Err(DataFusionError::Internal(
                    "Unsupported data type in hasher".to_string(),
                )));
                false
            }
        });

    err.unwrap_or(Ok(res))
}

macro_rules! cmp_rows_elem {
    ($array_type:ident, $l: ident, $r: ident, $left: ident, $right: ident) => {{
        let left_array = $l.as_any().downcast_ref::<$array_type>().unwrap();
        let right_array = $r.as_any().downcast_ref::<$array_type>().unwrap();

        match (left_array.is_null($left), right_array.is_null($right)) {
            (false, false) => {
                let cmp = left_array
                    .value($left)
                    .partial_cmp(&right_array.value($right))?;
                if cmp != Ordering::Equal {
                    res = cmp;
                    break;
                }
            }
            _ => unreachable!(),
        }
    }};
}

/// compare left row with right row
fn comp_rows(
    left: usize,
    right: usize,
    left_arrays: &[ArrayRef],
    right_arrays: &[ArrayRef],
) -> Result<Ordering> {
    let mut res = Ordering::Equal;
    for (l, r) in left_arrays.iter().zip(right_arrays) {
        match l.data_type() {
            DataType::Null => {}
            DataType::Boolean => cmp_rows_elem!(BooleanArray, l, r, left, right),
            DataType::Int8 => cmp_rows_elem!(Int8Array, l, r, left, right),
            DataType::Int16 => cmp_rows_elem!(Int16Array, l, r, left, right),
            DataType::Int32 => cmp_rows_elem!(Int32Array, l, r, left, right),
            DataType::Int64 => cmp_rows_elem!(Int64Array, l, r, left, right),
            DataType::UInt8 => cmp_rows_elem!(UInt8Array, l, r, left, right),
            DataType::UInt16 => cmp_rows_elem!(UInt16Array, l, r, left, right),
            DataType::UInt32 => cmp_rows_elem!(UInt32Array, l, r, left, right),
            DataType::UInt64 => cmp_rows_elem!(UInt64Array, l, r, left, right),
            DataType::Timestamp(_, None) => {
                cmp_rows_elem!(Int64Array, l, r, left, right)
            }
            DataType::Utf8 => cmp_rows_elem!(StringArray, l, r, left, right),
            DataType::LargeUtf8 => cmp_rows_elem!(LargeStringArray, l, r, left, right),
            _ => {
                // This is internal because we should have caught this before.
                return Err(DataFusionError::Internal(
                    "Unsupported data type in sort merge join comparator".to_string(),
                ));
            }
        }
    }

    Ok(res)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::TryExtend;
    use arrow::array::{MutableDictionaryArray, MutableUtf8Array};

    use crate::physical_plan::joins::check_join_set_is_valid;

    use super::*;

    fn check(left: &[Column], right: &[Column], on: &[(Column, Column)]) -> Result<()> {
        let left = left
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        let right = right
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        check_join_set_is_valid(&left, &right, on)
    }

    #[test]
    fn check_valid() -> Result<()> {
        let left = vec![Column::new("a", 0), Column::new("b1", 1)];
        let right = vec![Column::new("a", 0), Column::new("b2", 1)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        check(&left, &right, on)?;
        Ok(())
    }

    #[test]
    fn check_not_in_right() {
        let left = vec![Column::new("a", 0), Column::new("b", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_not_in_left() {
        let left = vec![Column::new("b", 0)];
        let right = vec![Column::new("a", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_collision() {
        // column "a" would appear both in left and right
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("a", 0), Column::new("b", 1)];
        let on = &[(Column::new("a", 0), Column::new("b", 1))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_in_right() {
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("b", 0))];

        assert!(check(&left, &right, on).is_ok());
    }
}
