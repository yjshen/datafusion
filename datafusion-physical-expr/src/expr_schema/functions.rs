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

//! Declaration of built-in (scalar) functions.
//! This module contains built-in functions' enumeration and metadata.
//!
//! Generally, a function has:
//! * a signature
//! * a return type, that is a function of the incoming argument's types
//! * the computation, that must accept each valid signature
//!
//! * Signature: see `Signature`
//! * Return type: a function `(arg_types) -> return_type`. E.g. for sqrt, ([f32]) -> f32, ([f64]) -> f64.
//!
//! This module also has a set of coercion rules to improve user experience: if an argument i32 is passed
//! to a function that supports f64, it is coerced to f64.

use arrow::datatypes::{DataType, Field, TimeUnit};
use datafusion_common::{DataFusionError, Result};

use crate::array_expressions;
use crate::coercion_rule::type_coercion::data_types;
use crate::expressions::SUPPORTED_NULLIF_TYPES;
use datafusion_expr::{BuiltinScalarFunction, Signature, TypeSignature};

macro_rules! make_utf8_to_return_type {
    ($FUNC:ident, $largeUtf8Type:expr, $utf8Type:expr) => {
        fn $FUNC(arg_type: &DataType, name: &str) -> Result<DataType> {
            Ok(match arg_type {
                DataType::LargeUtf8 => $largeUtf8Type,
                DataType::Utf8 => $utf8Type,
                _ => {
                    // this error is internal as `data_types` should have captured this.
                    return Err(DataFusionError::Internal(format!(
                        "The {:?} function can only accept strings.",
                        name
                    )));
                }
            })
        }
    };
}

make_utf8_to_return_type!(utf8_to_str_type, DataType::LargeUtf8, DataType::Utf8);
make_utf8_to_return_type!(utf8_to_int_type, DataType::Int64, DataType::Int32);
make_utf8_to_return_type!(utf8_to_binary_type, DataType::Binary, DataType::Binary);

/// Returns the datatype of the scalar function
pub fn return_type(
    fun: &BuiltinScalarFunction,
    input_expr_types: &[DataType],
) -> Result<DataType> {
    // Note that this function *must* return the same type that the respective physical expression returns
    // or the execution panics.

    if input_expr_types.is_empty() && !fun.supports_zero_argument() {
        return Err(DataFusionError::Internal(format!(
            "Builtin scalar function {} does not support empty arguments",
            fun
        )));
    }

    // verify that this is a valid set of data types for this function
    data_types(input_expr_types, &signature(fun))?;

    // the return type of the built in function.
    // Some built-in functions' return type depends on the incoming type.
    match fun {
        BuiltinScalarFunction::Array => Ok(DataType::FixedSizeList(
            Box::new(Field::new("item", input_expr_types[0].clone(), true)),
            input_expr_types.len() as i32,
        )),
        BuiltinScalarFunction::Ascii => Ok(DataType::Int32),
        BuiltinScalarFunction::BitLength => {
            utf8_to_int_type(&input_expr_types[0], "bit_length")
        }
        BuiltinScalarFunction::Btrim => utf8_to_str_type(&input_expr_types[0], "btrim"),
        BuiltinScalarFunction::CharacterLength => {
            utf8_to_int_type(&input_expr_types[0], "character_length")
        }
        BuiltinScalarFunction::Chr => Ok(DataType::Utf8),
        BuiltinScalarFunction::Concat => Ok(DataType::Utf8),
        BuiltinScalarFunction::ConcatWithSeparator => Ok(DataType::Utf8),
        BuiltinScalarFunction::DatePart => Ok(DataType::Int32),
        BuiltinScalarFunction::DateTrunc => {
            Ok(DataType::Timestamp(TimeUnit::Nanosecond, None))
        }
        BuiltinScalarFunction::InitCap => {
            utf8_to_str_type(&input_expr_types[0], "initcap")
        }
        BuiltinScalarFunction::Left => utf8_to_str_type(&input_expr_types[0], "left"),
        BuiltinScalarFunction::Lower => utf8_to_str_type(&input_expr_types[0], "lower"),
        BuiltinScalarFunction::Lpad => utf8_to_str_type(&input_expr_types[0], "lpad"),
        BuiltinScalarFunction::Ltrim => utf8_to_str_type(&input_expr_types[0], "ltrim"),
        BuiltinScalarFunction::MD5 => utf8_to_str_type(&input_expr_types[0], "md5"),
        BuiltinScalarFunction::NullIf => {
            // NULLIF has two args and they might get coerced, get a preview of this
            let coerced_types = data_types(input_expr_types, &signature(fun));
            coerced_types.map(|typs| typs[0].clone())
        }
        BuiltinScalarFunction::OctetLength => {
            utf8_to_int_type(&input_expr_types[0], "octet_length")
        }
        BuiltinScalarFunction::Random => Ok(DataType::Float64),
        BuiltinScalarFunction::RegexpReplace => {
            utf8_to_str_type(&input_expr_types[0], "regex_replace")
        }
        BuiltinScalarFunction::Repeat => utf8_to_str_type(&input_expr_types[0], "repeat"),
        BuiltinScalarFunction::Replace => {
            utf8_to_str_type(&input_expr_types[0], "replace")
        }
        BuiltinScalarFunction::Reverse => {
            utf8_to_str_type(&input_expr_types[0], "reverse")
        }
        BuiltinScalarFunction::Right => utf8_to_str_type(&input_expr_types[0], "right"),
        BuiltinScalarFunction::Rpad => utf8_to_str_type(&input_expr_types[0], "rpad"),
        BuiltinScalarFunction::Rtrim => utf8_to_str_type(&input_expr_types[0], "rtrimp"),
        BuiltinScalarFunction::SHA224 => {
            utf8_to_binary_type(&input_expr_types[0], "sha224")
        }
        BuiltinScalarFunction::SHA256 => {
            utf8_to_binary_type(&input_expr_types[0], "sha256")
        }
        BuiltinScalarFunction::SHA384 => {
            utf8_to_binary_type(&input_expr_types[0], "sha384")
        }
        BuiltinScalarFunction::SHA512 => {
            utf8_to_binary_type(&input_expr_types[0], "sha512")
        }
        BuiltinScalarFunction::Digest => {
            utf8_to_binary_type(&input_expr_types[0], "digest")
        }
        BuiltinScalarFunction::SplitPart => {
            utf8_to_str_type(&input_expr_types[0], "split_part")
        }
        BuiltinScalarFunction::StartsWith => Ok(DataType::Boolean),
        BuiltinScalarFunction::Strpos => utf8_to_int_type(&input_expr_types[0], "strpos"),
        BuiltinScalarFunction::Substr => utf8_to_str_type(&input_expr_types[0], "substr"),
        BuiltinScalarFunction::ToHex => Ok(match input_expr_types[0] {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                DataType::Utf8
            }
            _ => {
                // this error is internal as `data_types` should have captured this.
                return Err(DataFusionError::Internal(
                    "The to_hex function can only accept integers.".to_string(),
                ));
            }
        }),
        BuiltinScalarFunction::ToTimestamp => {
            Ok(DataType::Timestamp(TimeUnit::Nanosecond, None))
        }
        BuiltinScalarFunction::ToTimestampMillis => {
            Ok(DataType::Timestamp(TimeUnit::Millisecond, None))
        }
        BuiltinScalarFunction::ToTimestampMicros => {
            Ok(DataType::Timestamp(TimeUnit::Microsecond, None))
        }
        BuiltinScalarFunction::ToTimestampSeconds => {
            Ok(DataType::Timestamp(TimeUnit::Second, None))
        }
        BuiltinScalarFunction::Now => Ok(DataType::Timestamp(
            TimeUnit::Nanosecond,
            Some("UTC".to_owned()),
        )),
        BuiltinScalarFunction::Translate => {
            utf8_to_str_type(&input_expr_types[0], "translate")
        }
        BuiltinScalarFunction::Trim => utf8_to_str_type(&input_expr_types[0], "trim"),
        BuiltinScalarFunction::Upper => utf8_to_str_type(&input_expr_types[0], "upper"),
        BuiltinScalarFunction::RegexpMatch => Ok(match input_expr_types[0] {
            DataType::LargeUtf8 => {
                DataType::List(Box::new(Field::new("item", DataType::LargeUtf8, true)))
            }
            DataType::Utf8 => {
                DataType::List(Box::new(Field::new("item", DataType::Utf8, true)))
            }
            _ => {
                // this error is internal as `data_types` should have captured this.
                return Err(DataFusionError::Internal(
                    "The regexp_extract function can only accept strings.".to_string(),
                ));
            }
        }),

        BuiltinScalarFunction::Abs
        | BuiltinScalarFunction::Acos
        | BuiltinScalarFunction::Asin
        | BuiltinScalarFunction::Atan
        | BuiltinScalarFunction::Ceil
        | BuiltinScalarFunction::Cos
        | BuiltinScalarFunction::Exp
        | BuiltinScalarFunction::Floor
        | BuiltinScalarFunction::Log
        | BuiltinScalarFunction::Ln
        | BuiltinScalarFunction::Log10
        | BuiltinScalarFunction::Log2
        | BuiltinScalarFunction::Round
        | BuiltinScalarFunction::Signum
        | BuiltinScalarFunction::Sin
        | BuiltinScalarFunction::Sqrt
        | BuiltinScalarFunction::Tan
        | BuiltinScalarFunction::Trunc => match input_expr_types[0] {
            DataType::Float32 => Ok(DataType::Float32),
            _ => Ok(DataType::Float64),
        },
    }
}

/// the signatures supported by the function `fun`.
pub fn signature(fun: &BuiltinScalarFunction) -> Signature {
    // note: the physical expression must accept the type returned by this function or the execution panics.

    // for now, the list is small, as we do not have many built-in functions.
    match fun {
        BuiltinScalarFunction::Array => Signature::variadic(
            array_expressions::SUPPORTED_ARRAY_TYPES.to_vec(),
            fun.volatility(),
        ),
        BuiltinScalarFunction::Concat | BuiltinScalarFunction::ConcatWithSeparator => {
            Signature::variadic(vec![DataType::Utf8], fun.volatility())
        }
        BuiltinScalarFunction::Ascii
        | BuiltinScalarFunction::BitLength
        | BuiltinScalarFunction::CharacterLength
        | BuiltinScalarFunction::InitCap
        | BuiltinScalarFunction::Lower
        | BuiltinScalarFunction::MD5
        | BuiltinScalarFunction::OctetLength
        | BuiltinScalarFunction::Reverse
        | BuiltinScalarFunction::SHA224
        | BuiltinScalarFunction::SHA256
        | BuiltinScalarFunction::SHA384
        | BuiltinScalarFunction::SHA512
        | BuiltinScalarFunction::Trim
        | BuiltinScalarFunction::Upper => Signature::uniform(
            1,
            vec![DataType::Utf8, DataType::LargeUtf8],
            fun.volatility(),
        ),
        BuiltinScalarFunction::Btrim
        | BuiltinScalarFunction::Ltrim
        | BuiltinScalarFunction::Rtrim => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::Chr | BuiltinScalarFunction::ToHex => {
            Signature::uniform(1, vec![DataType::Int64], fun.volatility())
        }
        BuiltinScalarFunction::Lpad | BuiltinScalarFunction::Rpad => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64]),
                TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::Int64]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Utf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::Int64,
                    DataType::Utf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::LargeUtf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::Int64,
                    DataType::LargeUtf8,
                ]),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::Left
        | BuiltinScalarFunction::Repeat
        | BuiltinScalarFunction::Right => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64]),
                TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::Int64]),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::ToTimestamp => Signature::uniform(
            1,
            vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Timestamp(TimeUnit::Millisecond, None),
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Second, None),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::ToTimestampMillis => Signature::uniform(
            1,
            vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Second, None),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::ToTimestampMicros => Signature::uniform(
            1,
            vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Millisecond, None),
                DataType::Timestamp(TimeUnit::Second, None),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::ToTimestampSeconds => Signature::uniform(
            1,
            vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::Digest => {
            Signature::exact(vec![DataType::Utf8, DataType::Utf8], fun.volatility())
        }
        BuiltinScalarFunction::DateTrunc => Signature::exact(
            vec![
                DataType::Utf8,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::DatePart => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Date64]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(TimeUnit::Second, None),
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                ]),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::SplitPart => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::Utf8,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::LargeUtf8,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::LargeUtf8,
                    DataType::Int64,
                ]),
            ],
            fun.volatility(),
        ),

        BuiltinScalarFunction::Strpos | BuiltinScalarFunction::StartsWith => {
            Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                    TypeSignature::Exact(vec![DataType::Utf8, DataType::LargeUtf8]),
                    TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::Utf8]),
                    TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::LargeUtf8]),
                ],
                fun.volatility(),
            )
        }

        BuiltinScalarFunction::Substr => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64]),
                TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::Int64]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::Int64,
                    DataType::Int64,
                ]),
            ],
            fun.volatility(),
        ),

        BuiltinScalarFunction::Replace | BuiltinScalarFunction::Translate => {
            Signature::one_of(
                vec![TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ])],
                fun.volatility(),
            )
        }
        BuiltinScalarFunction::RegexpReplace => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
            ],
            fun.volatility(),
        ),

        BuiltinScalarFunction::NullIf => {
            Signature::uniform(2, SUPPORTED_NULLIF_TYPES.to_vec(), fun.volatility())
        }
        BuiltinScalarFunction::RegexpMatch => Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::LargeUtf8, DataType::Utf8]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::LargeUtf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
            ],
            fun.volatility(),
        ),
        BuiltinScalarFunction::Random => Signature::exact(vec![], fun.volatility()),
        // math expressions expect 1 argument of type f64 or f32
        // priority is given to f64 because e.g. `sqrt(1i32)` is in IR (real numbers) and thus we
        // return the best approximation for it (in f64).
        // We accept f32 because in this case it is clear that the best approximation
        // will be as good as the number of digits in the number
        _ => Signature::uniform(
            1,
            vec![DataType::Float64, DataType::Float32],
            fun.volatility(),
        ),
    }
}
