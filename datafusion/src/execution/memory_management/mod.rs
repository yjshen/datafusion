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
use crate::execution::memory_management::memory_pool::{
    ConstraintExecutionMemoryPool, DummyExecutionMemoryPool, ExecutionMemoryPool,
};
use async_trait::async_trait;
use hashbrown::HashMap;
use log::{debug, info};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

static mut CONSUMER_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct MemoryManager {
    execution_pool: Arc<dyn ExecutionMemoryPool>,
    partition_memory_manager: Arc<Mutex<HashMap<usize, PartitionMemoryManager>>>,
}

impl MemoryManager {
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

    pub fn acquire_exec_memory(
        self: Arc<Self>,
        required: usize,
        consumer: &MemoryConsumerId,
    ) -> Result<usize> {
        let partition_id = consumer.partition_id;
        let mut all_managers = self.partition_memory_manager.lock().unwrap();
        let partition_manager = all_managers
            .entry(partition_id)
            .or_insert_with(|| PartitionMemoryManager::new(partition_id, self.clone()));
        partition_manager.acquire_exec_memory(required, consumer)
    }

    pub fn acquire_exec_pool_memory(
        &self,
        required: usize,
        consumer: &MemoryConsumerId,
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
        consumer: &MemoryConsumerId,
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
    memory_manager: Weak<MemoryManager>,
    partition_id: usize,
    consumers: Mutex<HashMap<MemoryConsumerId, usize>>,
}

impl PartitionMemoryManager {
    pub fn new(partition_id: usize, memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            memory_manager: Arc::downgrade(&memory_manager),
            partition_id,
            consumers: Mutex::new(HashMap::new()),
        }
    }

    pub fn acquire_exec_memory(
        &mut self,
        required: usize,
        consumer: &MemoryConsumerId,
    ) -> Result<usize> {
        let consumers = self.consumers.get_mut().unwrap();
        let got = self
            .memory_manager
            .upgrade()
            .ok_or_else(|| {
                DataFusionError::Execution("Failed to get MemoryManager".to_string())
            })?
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

        let entry = consumers.entry(consumer.clone()).or_insert(0);
        *entry += got;

        debug!("{} acquired {}", consumer, got);
        Ok(got)
    }

    pub fn show_memory_usage(&self) -> Result<()> {
        info!("Memory usage for partition {}", self.partition_id);
        let consumers = self.consumers.lock().unwrap();
        let mut used = 0;
        for (id, c) in consumers.iter() {
            let cur_used = *c;
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
    fn allocate(&self, required: usize) -> Result<()> {
        let got = self
            .memory_manager()
            .acquire_exec_memory(required, self.id())?;
        self.update_used(got as isize);
        Ok(())
    }
    /// Spill at least `size` bytes to disk and frees memory
    async fn spill(&mut self, size: usize, trigger: &dyn MemoryConsumer)
        -> Result<usize>;
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
