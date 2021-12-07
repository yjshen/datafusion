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

//! Defines the External shuffle repartition plan

use crate::error::{DataFusionError, Result};
use crate::execution::memory_management::{
    MemoryConsumer, MemoryConsumerId, MemoryManager,
};
use crate::execution::runtime_env::RuntimeEnv;
use crate::physical_plan::hash_utils::create_hashes;
use crate::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use crate::physical_plan::Partitioning;
use ahash::RandomState;
use arrow::array::*;
use arrow::compute::take;
use arrow::datatypes::{PhysicalType, PrimitiveType, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::mutable_record_batch::MutableRecordBatch;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::lock::Mutex;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::sync::Arc;
use log::{error, info};


#[derive(Default)]
struct PartitionBuffer {
    frozen: Vec<RecordBatch>,
    active: Option<MutableRecordBatch>,
}

struct DataIndex {
    data_file: String,
    index_file: String,
}

fn create_tmp_data_index(runtime: &Arc<RuntimeEnv>) -> Result<DataIndex> {
    let data_file = runtime.disk_manager.create_tmp_file()?;
    let index_file = runtime.disk_manager.create_tmp_file()?;
    Ok(DataIndex {
        data_file,
        index_file,
    })
}

macro_rules! append {
    ($TO:ty, $FROM:ty, $to: ident, $from: ident) => {{
        let to = $to.as_mut_any().downcast_mut::<$TO>().unwrap();
        let from = $from.as_any().downcast_ref::<$FROM>().unwrap();
        to.extend_trusted_len(from.into_iter());
    }};
}

fn append_column(to: &mut Box<dyn MutableArray>, from: &Arc<dyn Array>) {
    // output buffered start `buffered_idx`, len `rows_to_output`
    match to.data_type().to_physical_type() {
        PhysicalType::Boolean => append!(MutableBooleanArray, BooleanArray, to, from),
        PhysicalType::Primitive(primitive) => match primitive {
            PrimitiveType::Int8 => append!(Int8Vec, Int8Array, to, from),
            PrimitiveType::Int16 => append!(Int16Vec, Int16Array, to, from),
            PrimitiveType::Int32 => append!(Int32Vec, Int32Array, to, from),
            PrimitiveType::Int64 => append!(Int64Vec, Int64Array, to, from),
            PrimitiveType::Float32 => append!(Float32Vec, Float32Array, to, from),
            PrimitiveType::Float64 => append!(Float64Vec, Float64Array, to, from),
            _ => todo!(),
        },
        PhysicalType::Utf8 => append!(MutableUtf8Array<i32>, Utf8Array<i32>, to, from),
        _ => todo!(),
    }
}

struct ShuffleRepartitioner {
    id: MemoryConsumerId,
    schema: SchemaRef,
    buffered_partitions: Mutex<Vec<PartitionBuffer>>,
    spills: Mutex<Vec<DataIndex>>,
    /// Sort expressions
    /// Partitioning scheme to use
    partitioning: Partitioning,
    num_output_partitions: usize,
    runtime: Arc<RuntimeEnv>,
    metrics: ExecutionPlanMetricsSet,
    used: AtomicIsize,
    spilled_bytes: AtomicUsize,
    spilled_count: AtomicUsize,
    insert_finished: AtomicBool,
    random: RandomState,
}

impl ShuffleRepartitioner {
    pub fn new(
        partition_id: usize,
        schema: SchemaRef,
        partitioning: Partitioning,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        let num_output_partitions = partitioning.partition_count();
        Self {
            id: MemoryConsumerId::new(partition_id),
            schema,
            buffered_partitions: Mutex::new(Default::default()),
            spills: Mutex::new(vec![]),
            partitioning,
            num_output_partitions,
            runtime,
            metrics: ExecutionPlanMetricsSet::new(),
            used: AtomicIsize::new(0),
            spilled_bytes: AtomicUsize::new(0),
            spilled_count: AtomicUsize::new(0),
            insert_finished: AtomicBool::new(false),
            random: RandomState::with_seeds(0, 0, 0, 0),
        }
    }

    pub(crate) fn finish_insert(&self) {
        self.insert_finished.store(true, Ordering::SeqCst);
    }

    async fn spill_while_inserting(&self) -> Result<usize> {
        info!(
            "{} spilling shuffle data of {} to disk while inserting ({} time(s) so far)",
            self.str_repr(),
            self.get_used(),
            self.spilled_count()
        );

        let partition = self.partition_id();
        let mut buffered_partitions = self.buffered_partitions.lock().await;
        // we could always get a chance to free some memory as long as we are holding some
        if buffered_partitions.len() == 0 {
            return Ok(0);
        }

        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let data_n_index = create_tmp_data_index(&self.runtime)?;

        let path = self.runtime.disk_manager.create_tmp_file()?;
        let total_size = spill_into(
            &mut *buffered_partitions,
            self.schema.clone(),
            &data_n_index,
        )
        .await;

        let total_size = spill(&mut stream?, path.clone(), self.schema.clone()).await?;

        let mut spills = self.spills.lock().await;
        self.spilled_count.fetch_add(1, Ordering::SeqCst);
        self.spilled_bytes.fetch_add(total_size, Ordering::SeqCst);
        spills.push(data_n_index);
        Ok(total_size)
    }

    async fn insert_batch(&self, input: RecordBatch) -> Result<()> {
        let random_state = self.random.clone();
        let num_output_partitions = self.num_output_partitions;
        let shuffle_batch_size = self.runtime.shuffle_batch_size();
        match &self.partitioning {
            Partitioning::Hash(exprs, _) => {
                let hashes_buf = &mut vec![];
                let arrays = exprs
                    .iter()
                    .map(|expr| {
                        Ok(expr
                            .evaluate(&input)?
                            .into_array(input.num_rows()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                hashes_buf.resize(arrays[0].len(), 0);
                // Hash arrays and compute buckets based on number of partitions
                let hashes = create_hashes(&arrays, &random_state, hashes_buf)?;
                let mut indices = vec![vec![]; num_output_partitions];
                for (index, hash) in hashes.iter().enumerate() {
                    indices[(*hash % num_output_partitions as u64) as usize]
                        .push(index as u64)
                }

                for (num_output_partition, partition_indices) in
                    indices.into_iter().enumerate()
                {
                    let mut buffered_partitions = self.buffered_partitions.lock().await;
                    let output = &mut buffered_partitions[num_output_partition];
                    let indices = UInt64Array::from_slice(&partition_indices);
                    // Produce batches based on indices
                    let columns = input
                        .columns()
                        .iter()
                        .map(|c| {
                            take::take(c.as_ref(), &indices)
                                .map(|x| x.into())
                                .map_err(|e| DataFusionError::Execution(e.to_string()))
                        })
                        .collect::<Result<Vec<Arc<dyn Array>>>>()?;

                    if partition_indices.len() > shuffle_batch_size {
                        let output_batch =
                            RecordBatch::try_new(input.schema().clone(), columns)?;
                        output.frozen.push(output_batch);
                    } else {
                        match output.active {
                            None => {
                                let buffer = MutableRecordBatch::new(
                                    shuffle_batch_size,
                                    self.schema.clone(),
                                )
                                .map_err(DataFusionError::ArrowError)?;
                                output.active = Some(buffer);
                            }
                            _ => {}
                        }

                        let mut batch = output.active.take().unwrap();
                        batch
                            .arrays
                            .iter_mut()
                            .zip(columns.iter())
                            .for_each(|(to, from)| append_column(to, from));
                        batch.append(partition_indices.len());

                        if batch.is_full() {
                            let result = batch.output_and_reset()?;
                            output.frozen.push(result);
                        }
                        output.active = Some(batch);
                    }
                }
            }
            other => {
                // this should be unreachable as long as the validation logic
                // in the constructor is kept up-to-date
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported repartitioning scheme {:?}",
                    other
                )));
            }
        }
        Ok(())
    }
}

/// consume the `sorted_bathes` and do in_mem_sort
async fn spill_into(
    buffered_partitions: &mut Vec<PartitionBuffer>,
    schema: SchemaRef,
    data_n_index: &DataIndex,
) -> Result<()> {
    Ok(())
}

impl Debug for ShuffleRepartitioner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalSorter")
            .field("id", &self.id())
            .field("memory_used", &self.get_used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spilled_count", &self.spilled_count())
            .finish()
    }
}

#[async_trait]
impl MemoryConsumer for ShuffleRepartitioner {
    fn name(&self) -> String {
        "ShuffleRepartitioner".to_owned()
    }

    fn id(&self) -> &MemoryConsumerId {
        &self.id
    }

    fn memory_manager(&self) -> Arc<MemoryManager> {
        self.runtime.memory_manager.clone()
    }

    async fn spill_inner(
        &self,
        size: usize,
        trigger: &MemoryConsumerId,
    ) -> Result<usize> {
        if !self.insert_finished.load(Ordering::SeqCst) {
            let total_size = self.spill_while_inserting().await;
            total_size
        } else {
            Ok(0)
        }
    }

    fn get_used(&self) -> isize {
        self.used.load(Ordering::SeqCst)
    }

    fn update_used(&self, delta: isize) {
        self.used.fetch_add(delta, Ordering::SeqCst);
    }

    fn spilled_bytes(&self) -> usize {
        self.spilled_bytes.load(Ordering::SeqCst)
    }

    fn spilled_bytes_add(&self, add: usize) {
        self.spilled_bytes.fetch_add(add, Ordering::SeqCst);
    }

    fn spilled_count(&self) -> usize {
        self.spilled_count.load(Ordering::SeqCst)
    }

    fn spilled_count_increment(&self) {
        self.spilled_count.fetch_add(1, Ordering::SeqCst);
    }
}
