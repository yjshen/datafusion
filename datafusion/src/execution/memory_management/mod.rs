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

pub mod memory_pool;

use crate::error::DataFusionError::OutOfMemory;
use crate::error::{DataFusionError, Result};
use crate::execution::disk_manager::DiskManager;
use crate::execution::memory_management::memory_pool::{
    ConstraintExecutionMemoryPool, DummyExecutionMemoryPool, ExecutionMemoryPool,
};
use async_trait::async_trait;
use hashbrown::{HashMap, HashSet};
use log::{debug, info};
use parking_lot::Mutex;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static mut CONSUMER_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct MemoryManager {
    execution_pool: Arc<dyn ExecutionMemoryPool>,
    partition_memory_manager: Arc<Mutex<HashMap<usize, PartitionMemoryManager>>>,
}

impl MemoryManager {
    pub fn new(exec_pool_size: usize) -> Self {
        let pool: dyn ExecutionMemoryPool = if exec_pool_size == usize::MAX {
            DummyExecutionMemoryPool::new()
        } else {
            ConstraintExecutionMemoryPool::new(exec_pool_size)
        };
        Self {
            execution_pool: Arc::new(pool),
            partition_memory_manager: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn acquire_exec_memory(
        self: Arc<Self>,
        required: usize,
        consumer: &dyn MemoryConsumer,
    ) -> Result<usize> {
        let partition_id = consumer.partition_id();
        let partition_manager = {
            let mut all_managers = self.partition_memory_manager.lock();
            all_managers.entry(partition_id).or_insert_with(|| {
                PartitionMemoryManager::new(partition_id, self.clone())
            })
        };
        partition_manager.acquire_exec_memory(required, consumer)
    }

    pub fn acquire_exec_pool_memory(
        &self,
        required: usize,
        consumer: &dyn MemoryConsumer,
    ) -> usize {
        self.execution_pool.acquire_memory(required, consumer)
    }

    pub fn release_exec_pool_memory(&self, release_size: usize, partition_id: usize) {
        self.execution_pool
            .release_memory(release_size, partition_id)
    }

    pub fn update_exec_pool_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        consumer: &dyn MemoryConsumer,
    ) {
        self.execution_pool
            .update_usage(granted_size, real_size, consumer)
    }

    pub fn release_all_exec_pool_for_partition(&self, partition_id: usize) -> usize {
        self.execution_pool.release_all(partition_id)
    }

    pub fn exec_memory_used(&self) -> usize {
        self.execution_pool.memory_used()
    }

    pub fn exec_memory_used_for_partition(&self, partition_id: usize) -> usize {
        self.execution_pool.memory_used_partition(partition_id)
    }
}

fn next_id() -> usize {
    unsafe { CONSUMER_ID.fetch_add(1, Ordering::SeqCst) }
}

pub struct PartitionMemoryManager {
    memory_manager: Arc<MemoryManager>,
    partition_id: usize,
    consumers: Arc<Mutex<HashSet<dyn MemoryConsumer>>>,
}

impl PartitionMemoryManager {
    pub fn new(partition_id: usize, memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            memory_manager,
            partition_id,
            consumers: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn acquire_exec_memory(
        &mut self,
        required: usize,
        consumer: &dyn MemoryConsumer,
    ) -> Result<usize> {
        let mut consumers = self.consumers.lock();
        let mut got = self
            .memory_manager
            .acquire_exec_pool_memory(required, consumer);
        if got < required {
            // spill others first
        }

        if got < required {
            // spill itself
        }

        if got < required {
            return Err(OutOfMemory(format!(
                "Unable to acquire {} bytes of memory, got {}",
                required, got
            )));
        }

        consumers.insert(consumer);
        debug!("{} acquired {}", consumer.str_repr(), got);
        Ok(got)
    }

    pub fn show_memory_usage(&self) {
        info!("Memory usage for partition {}", self.partition_id);
        let mut consumers = self.consumers.lock();
        let mut used = 0;
        for c in consumers.iter() {
            let cur_used = c.get_used();
            used += cur_used;
            if cur_used > 0 {
                info!(
                    "Consumer {} acquired {}",
                    c.str_repr(),
                    human_readable_size(cur_used)
                )
            }
        }
        let no_consumer_size = self
            .memory_manager
            .exec_memory_used_for_partition(self.partition_id)
            - used;
        info!(
            "{} bytes of memory were used for partition {} without specific consumer",
            human_readable_size(no_consumer_size),
            self.partition_id
        )
    }
}

#[derive(Debug, Clone)]
pub struct MemoryConsumerId {
    partition_id: usize,
    id: usize,
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
pub trait MemoryConsumer {
    /// Display name of the consumer
    fn name(&self) -> String;
    /// Unique id of the consumer
    fn id(&self) -> &MemoryConsumerId;

    fn memory_manager(&self) -> Arc<MemoryManager>;
    /// partition that the consumer belongs to
    fn partition_id(&self) -> uszie {
        self.id().partition_id
    }
    /// Try allocate `required` bytes as needed
    fn allocate(&self, required: usize) -> Result<()> {
        let got = self.memory_manager().acquire_exec_memory(required, self)?;
        self.update_used(got as isize);
        Ok(())
    }
    /// Spill at least `size` bytes to disk and frees memory
    async fn spill(&self, size: usize, trigger: &dyn MemoryConsumer) -> Result<usize>;
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
            (size as f64 / TB, "TB")
        } else if size >= 2 * GB {
            (size as f64 / GB, "GB")
        } else if size >= 2 * MB {
            (size as f64 / MB, "MB")
        } else if size >= 2 * KB {
            (size as f64 / KB, "KB")
        } else {
            (size, "B")
        }
    };
    format!("{:.1} {}", value, unit)
}
