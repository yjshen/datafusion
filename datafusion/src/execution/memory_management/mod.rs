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

pub mod allocation_strategist;

use crate::error::DataFusionError::OutOfMemory;
use crate::error::{DataFusionError, Result};
use crate::execution::memory_management::allocation_strategist::{
    ConstraintEqualShareStrategist, DummyAllocationStrategist, MemoryAllocationStrategist,
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
    strategist: Arc<dyn MemoryAllocationStrategist>,
    partition_memory_manager: Arc<Mutex<HashMap<usize, PartitionMemoryManager>>>,
}

impl MemoryManager {
    /// Create memory manager based on configured execution pool size.
    pub fn new(exec_pool_size: usize) -> Self {
        let strategist: Arc<dyn MemoryAllocationStrategist> =
            if exec_pool_size == usize::MAX {
                Arc::new(DummyAllocationStrategist::new())
            } else {
                Arc::new(ConstraintEqualShareStrategist::new(exec_pool_size))
            };
        Self {
            strategist,
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
        self.strategist
            .acquire_memory(required, consumer.partition_id)
            .await
    }

    pub(crate) async fn release_exec_pool_memory(
        &self,
        release_size: usize,
        partition_id: usize,
    ) {
        self.strategist
            .release_memory(release_size, partition_id)
            .await
    }

    /// Revise pool usage while handling variable length data structure.
    /// In this case, we may estimate and allocate in advance, and revise the usage
    /// after the construction of the data structure.
    #[allow(dead_code)]
    pub(crate) async fn update_exec_pool_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        consumer: &MemoryConsumerId,
    ) {
        self.strategist
            .update_usage(granted_size, real_size, consumer)
            .await
    }

    /// Called during the shutdown procedure of a partition, for memory reclamation.
    #[allow(dead_code)]
    pub(crate) async fn release_all_exec_pool_for_partition(
        &self,
        partition_id: usize,
    ) -> usize {
        self.strategist.release_all(partition_id).await
    }

    #[allow(dead_code)]
    pub(crate) fn exec_memory_used(&self) -> usize {
        self.strategist.memory_used()
    }

    pub(crate) fn exec_memory_used_for_partition(&self, partition_id: usize) -> usize {
        self.strategist.memory_used_partition(partition_id)
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

    /// Try to acquire `required` of execution memory for the consumer and return the number of bytes
    /// obtained, or return OutOfMemoryError if no enough memory avaiable even after possible spills.
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
                        c.str_repr()
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
                    self.partition_id,
                    released,
                    consumer.str_repr()
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

    /// log current memory usage for all consumers in this partition
    pub async fn show_memory_usage(&self) -> Result<()> {
        info!("Memory usage for partition {}", self.partition_id);
        let consumers = self.consumers.lock().await;
        let mut used = 0;
        for (_, c) in consumers.iter() {
            let cur_used = c.get_used();
            used += cur_used;
            if cur_used > 0 {
                info!(
                    "Consumer {} acquired {}",
                    c.str_repr(),
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
/// Id that uniquely identifies a Memory Consumer
pub struct MemoryConsumerId {
    /// partition the consumer belongs to
    pub partition_id: usize,
    /// unique id
    pub id: usize,
}

impl MemoryConsumerId {
    /// Auto incremented new Id
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
/// A memory consumer that supports spilling.
pub trait MemoryConsumer: Send + Sync + Debug {
    /// Display name of the consumer
    fn name(&self) -> String;

    /// Unique id of the consumer
    fn id(&self) -> &MemoryConsumerId;

    /// Ptr to MemoryManager
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

    /// Spill at least `size` bytes to disk and update related counters
    async fn spill(&self, size: usize, trigger: &MemoryConsumerId) -> Result<usize> {
        let released = self.spill_inner(size, trigger).await?;
        if released > 0 {
            self.memory_manager()
                .release_exec_pool_memory(released, self.id().partition_id)
                .await;
            self.update_used(-(released as isize));
            self.spilled_bytes_add(released);
            self.spilled_count_increment();
        }
        Ok(released)
    }

    /// Spill at least `size` bytes to disk and frees memory
    async fn spill_inner(&self, size: usize, trigger: &MemoryConsumerId)
        -> Result<usize>;

    /// Get current memory usage for the consumer itself
    fn get_used(&self) -> isize;

    /// Update memory usage
    fn update_used(&self, delta: isize);

    /// Get total number of spilled bytes so far
    fn spilled_bytes(&self) -> usize;

    /// Update spilled bytes counter
    fn spilled_bytes_add(&self, add: usize);

    /// Get total number of triggered spills so far
    fn spilled_count(&self) -> usize;

    /// Update spilled count
    fn spilled_count_increment(&self);

    /// String representation for the consumer
    fn str_repr(&self) -> String {
        format!("{}({})", self.name(), self.id())
    }

    #[inline]
    /// log during spilling
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
