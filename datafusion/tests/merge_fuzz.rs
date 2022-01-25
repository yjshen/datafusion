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

//! Tests various configurations of merging streams
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int32Array},
    compute::SortOptions,
    record_batch::RecordBatch,
};
use datafusion::{
    execution::runtime_env::{RuntimeConfig, RuntimeEnv},
    physical_plan::{
        collect,
        expressions::{col, PhysicalSortExpr},
        memory::MemoryExec,
        sorts::sort_preserving_merge::SortPreservingMergeExec,
    },
};
use rand::{prelude::StdRng, Rng, SeedableRng};

/// Defines each test case
struct TestCase {
    name: &'static str,
    input: Vec<Vec<RecordBatch>>,
}

#[tokio::test]
async fn test_merge_2() {
    TestCase {
        name: "2x sorted numbers 0 - 99",
        input: vec![
            make_staggered_batches(0, 100, 2),
            make_staggered_batches(0, 100, 3),
        ],
    }
    .run()
    .await;
}

#[tokio::test]
async fn test_merge_2_no_overlap() {
    TestCase {
        name: "0..20 and then 20..40",
        input: vec![
            make_staggered_batches(0, 20, 2),
            make_staggered_batches(20, 40, 3),
        ],
    }
    .run()
    .await;
}

#[tokio::test]
async fn test_merge_3() {
    TestCase {
        name: "2x sorted numbers 0 - 99, 1x 0 - 50",
        input: vec![
            make_staggered_batches(0, 100, 2),
            make_staggered_batches(0, 100, 3),
            make_staggered_batches(0, 51, 4),
        ],
    }
    .run()
    .await;
}

impl TestCase {
    // runs the test case
    async fn run(self) {
        let TestCase { name, input } = self;

        for batch_size in vec![1, 2, 7, 49, 50, 51, 100] {
            let first_batch = input
                .iter()
                .map(|p| p.iter())
                .flatten()
                .next()
                .expect("at least one batch");
            let schema = first_batch.schema();

            let sort = vec![PhysicalSortExpr {
                expr: col("x", &schema).unwrap(),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }];

            let exec = MemoryExec::try_new(&input, schema, None).unwrap();
            let merge = Arc::new(SortPreservingMergeExec::new(sort, Arc::new(exec)));

            let runtime_config = RuntimeConfig::new().with_batch_size(batch_size);

            let runtime = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
            let collected = collect(merge, runtime).await.unwrap();

            let expected = partitions_to_sorted_vec(&input);
            let actual = batches_to_vec(&collected);

            assert_eq!(
                expected, actual,
                "failure in {} @ batch_size {}",
                name, batch_size
            );
        }
    }
}

/// Extracts the i32 values from the set of batches and returns them as a single Vec
fn batches_to_vec(batches: &[RecordBatch]) -> Vec<Option<i32>> {
    batches
        .iter()
        .map(|batch| {
            assert_eq!(batch.num_columns(), 1);
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
        })
        .flatten()
        .collect()
}

// extract values from batches and sort them
fn partitions_to_sorted_vec(partitions: &[Vec<RecordBatch>]) -> Vec<Option<i32>> {
    let mut values: Vec<_> = partitions
        .iter()
        .map(|batches| batches_to_vec(batches).into_iter())
        .flatten()
        .collect();

    values.sort_unstable();
    values
}

/// Return the values `low..high` in order, in randomly sized
/// record batches in a field named 'x' of type `Int32`
fn make_staggered_batches(low: i32, high: i32, seed: u64) -> Vec<RecordBatch> {
    let input: Int32Array = (low..high).map(Some).collect();

    // split into several record batches
    let mut remainder =
        RecordBatch::try_from_iter(vec![("x", Arc::new(input) as ArrayRef)]).unwrap();

    let mut batches = vec![];

    // use a random number generator to pick a random sized output
    let mut rng = StdRng::seed_from_u64(seed);
    while remainder.num_rows() > 0 {
        let batch_size = rng.gen_range(0..remainder.num_rows() + 1);

        batches.push(remainder.slice(0, batch_size));
        remainder = remainder.slice(batch_size, remainder.num_rows() - batch_size);
    }
    batches
}
