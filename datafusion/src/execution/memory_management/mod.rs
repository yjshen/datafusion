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

//! Manages all available memory during query execution

pub mod memory_pool;

use crate::error::DataFusionError::OutOfMemory;
use crate::error::{DataFusionError, Result};
use crate::execution::memory_management::memory_pool::{
    ConstraintExecutionMemoryPool, DummyExecutionMemoryPool, ExecutionMemoryPool,
};
use async_trait::async_trait;
use futures::lock::Mutex;
use hashbrown::HashMap;
use log::{debug, info};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

static mut CONSUMER_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
/// Memory manager that enforces how execution memory is shared between all kinds of memory consumers.
/// Execution memory refers to that used for computation in sorts, aggregations, joins and shuffles.
pub struct MemoryManager {
    execution_pool: Arc<dyn ExecutionMemoryPool>,
    partition_memory_manager: Arc<Mutex<HashMap<usize, PartitionMemoryManager>>>,
}

impl MemoryManager {
    /// Create memory manager based on configured execution pool size.
    pub fn new(exec_pool_size: usize) -> Self {
        let execution_pool: Arc<dyn ExecutionMemoryPool> = if exec_pool_size == usize::MAX
        {
            Arc::new(DummyExecutionMemoryPool::new())
        } else {
            Arc::new(ConstraintExecutionMemoryPool::new(exec_pool_size))
        };
        Self {
            execution_pool,
            partition_memory_manager: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Acquire size of `required` memory from manager
    pub async fn acquire_exec_memory(
        self: &Arc<Self>,
        required: usize,
        consumer_id: &MemoryConsumerId,
    ) -> Result<usize> {
        let partition_id = consumer_id.partition_id;
        let mut all_managers = self.partition_memory_manager.lock().await;
        let partition_manager = all_managers
            .entry(partition_id)
            .or_insert_with(|| PartitionMemoryManager::new(partition_id, self.clone()));
        partition_manager
            .acquire_exec_memory(required, consumer_id)
            .await
    }

    /// Register consumer to manager, for memory tracking and enables spilling by
    /// memory used.
    pub async fn register_consumer(self: &Arc<Self>, consumer: Arc<dyn MemoryConsumer>) {
        let partition_id = consumer.partition_id();
        let mut all_managers = self.partition_memory_manager.lock().await;
        let partition_manager = all_managers
            .entry(partition_id)
            .or_insert_with(|| PartitionMemoryManager::new(partition_id, self.clone()));
        partition_manager.register_consumer(consumer).await;
    }

    pub(crate) async fn acquire_exec_pool_memory(
        &self,
        required: usize,
        consumer: &MemoryConsumerId,
    ) -> usize {
        self.execution_pool.acquire_memory(required, consumer).await
    }

    pub(crate) async fn release_exec_pool_memory(
        &self,
        release_size: usize,
        partition_id: usize,
    ) {
        self.execution_pool
            .release_memory(release_size, partition_id)
            .await
    }

    pub(crate) async fn update_exec_pool_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        consumer: &MemoryConsumerId,
    ) {
        self.execution_pool
            .update_usage(granted_size, real_size, consumer)
            .await
    }

    pub(crate) async fn release_all_exec_pool_for_partition(
        &self,
        partition_id: usize,
    ) -> usize {
        self.execution_pool.release_all(partition_id).await
    }

    pub(crate) fn exec_memory_used(&self) -> usize {
        self.execution_pool.memory_used()
    }

    pub(crate) fn exec_memory_used_for_partition(&self, partition_id: usize) -> usize {
        self.execution_pool.memory_used_partition(partition_id)
    }
}

fn next_id() -> usize {
    unsafe { CONSUMER_ID.fetch_add(1, Ordering::SeqCst) }
}

/// Memory manager that tracks all consumers for a specific partition
/// Trigger the spill for consumer(s) when memory is insufficient
pub struct PartitionMemoryManager {
    memory_manager: Weak<MemoryManager>,
    partition_id: usize,
    consumers: Mutex<HashMap<MemoryConsumerId, Arc<dyn MemoryConsumer>>>,
}

impl PartitionMemoryManager {
    /// Create manager for a partition
    pub fn new(partition_id: usize, memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            memory_manager: Arc::downgrade(&memory_manager),
            partition_id,
            consumers: Mutex::new(HashMap::new()),
        }
    }

    /// Register a memory consumer at its first appearance
    pub async fn register_consumer(&self, consumer: Arc<dyn MemoryConsumer>) {
        let mut consumers = self.consumers.lock().await;
        let id = consumer.id().clone();
        consumers.insert(id, consumer);
    }

    pub async fn acquire_exec_memory(
        &self,
        required: usize,
        consumer_id: &MemoryConsumerId,
    ) -> Result<usize> {
        let mut consumers = self.consumers.lock().await;
        let memory_manager = self.memory_manager.upgrade().ok_or_else(|| {
            DataFusionError::Execution("Failed to get MemoryManager".to_string())
        })?;
        let mut got = memory_manager
            .acquire_exec_pool_memory(required, consumer_id)
            .await;
        if got < required {
            // Try to release memory from other consumers first
            // Sort the consumers according to their memory usage and spill from
            // consumer that holds the maximum memory, to reduce the total frequency of
            // spilling

            let mut all_consumers: Vec<Arc<dyn MemoryConsumer>> = vec![];
            for c in consumers.iter() {
                all_consumers.push(c.1.clone());
            }
            all_consumers.sort_by(|a, b| b.get_used().cmp(&a.get_used()));

            for c in all_consumers.iter_mut() {
                if c.id() == consumer_id {
                    continue;
                }

                let released = c.spill(required - got, consumer_id).await?;
                if released > 0 {
                    debug!(
                        "Partition {} released {} from consumer {}",
                        self.partition_id,
                        released,
                        c.id()
                    );
                    got += memory_manager
                        .acquire_exec_pool_memory(required - got, consumer_id)
                        .await;
                    if got > required {
                        break;
                    }
                }
            }
        }

        if got < required {
            // spill itself
            let consumer = consumers.get_mut(consumer_id).unwrap();
            let released = consumer.spill(required - got, consumer_id).await?;
            if released > 0 {
                debug!(
                    "Partition {} released {} from consumer itself {}",
                    self.partition_id, released, consumer_id
                );
                got += memory_manager
                    .acquire_exec_pool_memory(required - got, consumer_id)
                    .await;
            }
        }

        if got < required {
            return Err(OutOfMemory(format!(
                "Unable to acquire {} bytes of memory, got {}",
                required, got
            )));
        }

        debug!("{} acquired {}", consumer_id, got);
        Ok(got)
    }

    pub async fn show_memory_usage(&self) -> Result<()> {
        info!("Memory usage for partition {}", self.partition_id);
        let consumers = self.consumers.lock().await;
        let mut used = 0;
        for (id, c) in consumers.iter() {
            let cur_used = c.get_used();
            used += cur_used;
            if cur_used > 0 {
                info!(
                    "Consumer {} acquired {}",
                    id,
                    human_readable_size(cur_used as usize)
                )
            }
        }
        let no_consumer_size = self
            .memory_manager
            .upgrade()
            .ok_or_else(|| {
                DataFusionError::Execution("Failed to get MemoryManager".to_string())
            })?
            .exec_memory_used_for_partition(self.partition_id)
            - (used as usize);
        info!(
            "{} bytes of memory were used for partition {} without specific consumer",
            human_readable_size(no_consumer_size),
            self.partition_id
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct MemoryConsumerId {
    pub partition_id: usize,
    pub id: usize,
}

impl MemoryConsumerId {
    pub fn new(partition_id: usize) -> Self {
        let id = next_id();
        Self { partition_id, id }
    }
}

impl Display for MemoryConsumerId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.partition_id, self.id)
    }
}

#[async_trait]
pub trait MemoryConsumer: Send + Sync + Debug {
    /// Display name of the consumer
    fn name(&self) -> String;
    /// Unique id of the consumer
    fn id(&self) -> &MemoryConsumerId;

    fn memory_manager(&self) -> Arc<MemoryManager>;
    /// partition that the consumer belongs to
    fn partition_id(&self) -> usize {
        self.id().partition_id
    }
    /// Try allocate `required` bytes as needed
    async fn allocate(&self, required: usize) -> Result<()> {
        let got = self
            .memory_manager()
            .acquire_exec_memory(required, self.id())
            .await?;
        self.update_used(got as isize);
        Ok(())
    }
    /// Spill at least `size` bytes to disk and frees memory
    async fn spill(&self, size: usize, trigger: &MemoryConsumerId) -> Result<usize>;
    /// Get current memory usage for the consumer itself
    fn get_used(&self) -> isize;

    fn update_used(&self, delta: isize);
    /// Get total number of spilled bytes so far
    fn spilled_bytes(&self) -> usize;
    /// Get total number of triggered spills so far
    fn spilled_count(&self) -> usize;

    fn str_repr(&self) -> String {
        format!("{}({})", self.name(), self.id())
    }

    #[inline]
    fn log_spill(&self, size: usize) {
        info!(
            "{} spilling of {} bytes to disk ({} times so far)",
            self.str_repr(),
            size,
            self.spilled_count()
        );
    }
}

const TB: u64 = 1 << 40;
const GB: u64 = 1 << 30;
const MB: u64 = 1 << 20;
const KB: u64 = 1 << 10;

fn human_readable_size(size: usize) -> String {
    let size = size as u64;
    let (value, unit) = {
        if size >= 2 * TB {
            (size as f64 / TB as f64, "TB")
        } else if size >= 2 * GB {
            (size as f64 / GB as f64, "GB")
        } else if size >= 2 * MB {
            (size as f64 / MB as f64, "MB")
        } else if size >= 2 * KB {
            (size as f64 / KB as f64, "KB")
        } else {
            (size as f64, "B")
        }
    };
    format!("{:.1} {}", value, unit)
}
