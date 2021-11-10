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

//! Execution Memory Pool that guarantees a memory allocation strategy

use crate::execution::memory_management::MemoryConsumerId;
use async_trait::async_trait;
use hashbrown::HashMap;
use log::{info, warn};
use std::cmp::min;
use std::fmt;
use std::fmt::{Debug, Formatter};
use tokio::runtime::Handle;
use tokio::sync::{Notify, RwLock};

#[async_trait]
pub(crate) trait MemoryAllocationStrategist: Sync + Send + Debug {
    /// Total memory available, which is pool_size - memory_used()
    fn memory_available(&self) -> usize;
    /// Current memory used by all PartitionManagers
    fn memory_used(&self) -> usize;
    /// Memory usage for a specific partition
    fn memory_used_partition(&self, partition_id: usize) -> usize;
    /// Acquire memory from a partition
    async fn acquire_memory(&self, required: usize, partition_id: usize) -> usize;
    /// Update memory usage for a partition
    async fn update_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        partition_id: usize,
    );
    /// release memory from partition
    async fn release_memory(&self, release_size: usize, partition_id: usize);
    /// release all memory acquired by a partition
    async fn release_all(&self, partition_id: usize) -> usize;
}

pub(crate) struct DummyAllocationStrategist {
    pool_size: usize,
}

impl DummyAllocationStrategist {
    pub fn new() -> Self {
        Self {
            pool_size: usize::MAX,
        }
    }
}

impl Debug for DummyAllocationStrategist {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DummyExecutionMemoryPool")
            .field("total", &self.pool_size)
            .finish()
    }
}

#[async_trait]
impl MemoryAllocationStrategist for DummyAllocationStrategist {
    fn memory_available(&self) -> usize {
        usize::MAX
    }

    fn memory_used(&self) -> usize {
        0
    }

    fn memory_used_partition(&self, _partition_id: usize) -> usize {
        0
    }

    async fn acquire_memory(&self, required: usize, _partition_id: usize) -> usize {
        required
    }

    async fn update_usage(
        &self,
        _granted_size: usize,
        _real_size: usize,
        _partition_id: usize,
    ) {
    }

    async fn release_memory(&self, _release_size: usize, _partition_id: usize) {}

    async fn release_all(&self, _partition_id: usize) -> usize {
        usize::MAX
    }
}

pub(crate) struct ConstraintEqualShareStrategist {
    pool_size: usize,
    /// memory usage per partition
    memory_usage: RwLock<HashMap<usize, usize>>,
    notify: Notify,
}

impl ConstraintEqualShareStrategist {
    pub fn new(size: usize) -> Self {
        Self {
            pool_size: size,
            memory_usage: RwLock::new(HashMap::new()),
            notify: Notify::new(),
        }
    }
}

impl Debug for ConstraintEqualShareStrategist {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintExecutionMemoryPool")
            .field("total", &self.pool_size)
            .field("used", &self.memory_used())
            .finish()
    }
}

#[async_trait]
impl MemoryAllocationStrategist for ConstraintEqualShareStrategist {
    fn memory_available(&self) -> usize {
        self.pool_size - self.memory_used()
    }

    fn memory_used(&self) -> usize {
        Handle::current()
            .block_on(async { self.memory_usage.read().await.values().sum() })
    }

    fn memory_used_partition(&self, partition_id: usize) -> usize {
        Handle::current().block_on(async {
            let partition_usage = self.memory_usage.read().await;
            match partition_usage.get(&partition_id) {
                None => 0,
                Some(v) => *v,
            }
        })
    }

    async fn acquire_memory(&self, required: usize, partition_id: usize) -> usize {
        assert!(required > 0);
        {
            let mut partition_usage = self.memory_usage.write().await;
            if !partition_usage.contains_key(&partition_id) {
                partition_usage.entry(partition_id).or_insert(0);
                // This will later cause waiting tasks to wake up and check numTasks again
                self.notify.notify_waiters();
            }
        }

        // Keep looping until we're either sure that we don't want to grant this request (because this
        // partition would have more than 1 / num_active_partition of the memory) or we have enough free
        // memory to give it (we always let each partition get at least 1 / (2 * num_active_partition)).
        loop {
            let partition_usage = self.memory_usage.read().await;
            let num_active_partition = partition_usage.len();
            let current_mem = *partition_usage.get(&partition_id).unwrap();

            let max_memory_per_partition = self.pool_size / num_active_partition;
            let min_memory_per_partition = self.pool_size / (2 * num_active_partition);

            // How much we can grant this partition; keep its share within 0 <= X <= 1 / num_active_partition
            let max_grant = match max_memory_per_partition.checked_sub(current_mem) {
                None => 0,
                Some(max_available) => min(required, max_available),
            };

            let total_used: usize = partition_usage.values().sum();
            let total_available = self.pool_size - total_used;
            // Only give it as much memory as is free, which might be none if it reached 1 / num_active_partition
            let to_grant = min(max_grant, total_available);

            // We want to let each task get at least 1 / (2 * numActiveTasks) before blocking;
            // if we can't give it this much now, wait for other tasks to free up memory
            // (this happens if older tasks allocated lots of memory before N grew)
            if to_grant < required && current_mem + to_grant < min_memory_per_partition {
                info!(
                    "{:?} waiting for at least 1/2N of pool to be free",
                    consumer
                );
                let _ = self.notify.notified().await;
            } else {
                drop(partition_usage);
                let mut partition_usage = self.memory_usage.write().await;
                *partition_usage.entry(partition_id).or_insert(0) += to_grant;
                return to_grant;
            }
        }
    }

    async fn update_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        partition_id: usize,
    ) {
        assert!(granted_size > 0);
        assert!(real_size > 0);
        if granted_size == real_size {
            return;
        } else {
            let mut partition_usage = self.memory_usage.write().await;
            if granted_size > real_size {
                *partition_usage.entry(consumer.partition_id).or_insert(0) -=
                    granted_size - real_size;
            } else {
                // TODO: this would have caused OOM already if size estimation ahead is much smaller than
                // that of actual allocation
                *partition_usage.entry(consumer.partition_id).or_insert(0) +=
                    real_size - granted_size;
            }
        }
    }

    async fn release_memory(&self, release_size: usize, partition_id: usize) {
        let partition_usage = self.memory_usage.read().await;
        let current_mem = match partition_usage.get(&partition_id) {
            None => 0,
            Some(v) => *v,
        };

        let to_free = if current_mem < release_size {
            warn!(
                "Release called to free {} but partition only holds {} from the pool",
                release_size, current_mem
            );
            current_mem
        } else {
            release_size
        };
        if partition_usage.contains_key(&partition_id) {
            drop(partition_usage);
            let mut partition_usage = self.memory_usage.write().await;
            let entry = partition_usage.entry(partition_id).or_insert(0);
            *entry -= to_free;
            if *entry == 0 {
                partition_usage.remove(&partition_id);
            }
        }
        self.notify.notify_waiters();
    }

    async fn release_all(&self, partition_id: usize) -> usize {
        let partition_usage = self.memory_usage.read().await;
        let mut current_mem = 0;
        match partition_usage.get(&partition_id) {
            None => return current_mem,
            Some(v) => current_mem = *v,
        }

        drop(partition_usage);
        let mut partition_usage = self.memory_usage.write().await;
        partition_usage.remove(&partition_id);
        self.notify.notify_waiters();
        return current_mem;
    }
}
