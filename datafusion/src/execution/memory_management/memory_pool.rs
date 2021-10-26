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

use crate::execution::memory_management::{MemoryConsumer, MemoryConsumerId};
use crate::physical_plan::aggregates::return_type;
use hashbrown::HashMap;
use log::{info, warn};
use std::cmp::{max, min};
use std::sync::{Arc, Condvar, Mutex};

pub(crate) trait ExecutionMemoryPool {
    fn memory_available(&self) -> usize;
    fn memory_used(&self) -> usize;
    fn memory_used_partition(&self, partition_id: usize) -> usize;
    fn acquire_memory(&self, required: usize, consumer: &dyn MemoryConsumer) -> usize;
    fn update_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        consumer: &dyn MemoryConsumer,
    );
    fn release_memory(&self, release_size: usize, partition_id: usize);
    fn release_all(&self, partition_id: usize) -> usize;
}

pub(crate) struct DummyExecutionMemoryPool {
    pool_size: usize,
}

impl DummyExecutionMemoryPool {
    pub fn new() -> Self {
        Self {
            pool_size: usize::MAX,
        }
    }
}

impl ExecutionMemoryPool for DummyExecutionMemoryPool {
    fn memory_available(&self) -> usize {
        usize::MAX
    }

    fn memory_used(&self) -> usize {
        0
    }

    fn memory_used_partition(&self, _partition_id: usize) -> usize {
        0
    }

    fn acquire_memory(&self, required: usize, _consumer: &dyn MemoryConsumer) -> usize {
        required
    }

    fn update_usage(
        &self,
        _granted_size: usize,
        _real_size: usize,
        _consumer: &dyn MemoryConsumer,
    ) {
    }

    fn release_memory(&self, _release_size: usize, _partition_id: usize) {}

    fn release_all(&self, _partition_id: usize) -> usize {
        usize::MAX
    }
}

pub(crate) struct ConstraintExecutionMemoryPool {
    pool_size: usize,
    /// memory usage per partition
    memory_usage: Mutex<HashMap<usize, usize>>,
    condvar: Condvar,
}

impl ConstraintExecutionMemoryPool {
    pub fn new(size: usize) -> Self {
        Self {
            pool_size: size,
            memory_usage: Mutex::new(HashMap::new()),
            condvar: Condvar::new(),
        }
    }
}

impl ExecutionMemoryPool for ConstraintExecutionMemoryPool {
    fn memory_available(&self) -> usize {
        self.pool_size - self.memory_used()
    }

    fn memory_used(&self) -> usize {
        let a = self.memory_usage.lock().unwrap();
        a.values().sum()
    }

    fn memory_used_partition(&self, partition_id: usize) -> usize {
        let partition_usage = self.memory_usage.lock().unwrap();
        partition_usage[partition_id].unwrap_or(0)
    }

    fn acquire_memory(&self, required: usize, consumer: &dyn MemoryConsumer) -> usize {
        assert!(required > 0);
        let partition_id = consumer.partition_id();
        let mut partition_usage = self.memory_usage.lock().unwrap();
        if !partition_usage.contains_key(&partition_id) {
            partition_usage.entry(partition_id).or_insert(0);
            self.condvar.notify_all();
        }

        // Keep looping until we're either sure that we don't want to grant this request (because this
        // partition would have more than 1 / num_active_partition of the memory) or we have enough free
        // memory to give it (we always let each partition get at least 1 / (2 * num_active_partition)).
        loop {
            let num_active_partition = partition_usage.len();
            let current_mem = *partition_usage.get(&partition_id).unwrap();

            let max_memory_per_partition = self.pool_size / num_active_partition;
            let min_memory_per_partition = self.pool_size / (2 * num_active_partition);

            // How much we can grant this partition; keep its share within 0 <= X <= 1 / num_active_partition
            let max_grant = match max_memory_per_partition.checked_sub(current_mem) {
                None => 0,
                Some(max_available) => min(required, max_available),
            };

            let total_used = partition_usage.values().sum();
            let total_available = self.pool_size - total_used;
            // Only give it as much memory as is free, which might be none if it reached 1 / num_active_partition
            let to_grant = min(max_grant, total_available);

            // We want to let each task get at least 1 / (2 * numActiveTasks) before blocking;
            // if we can't give it this much now, wait for other tasks to free up memory
            // (this happens if older tasks allocated lots of memory before N grew)
            if to_grant < required && current_mem + to_grant < min_memory_per_partition {
                info!("{} waiting for at least 1/2N of pool to be free", consumer);
                self.condvar.wait(&mut partition_usage);
            } else {
                *partition_usage.entry(partition_id).or_insert(0) += to_grant;
                return to_grant;
            }
        }
    }

    fn update_usage(
        &self,
        granted_size: usize,
        real_size: usize,
        consumer: &dyn MemoryConsumer,
    ) {
        assert!(granted_size > 0);
        assert!(real_size > 0);
        if granted_size == real_size {
            return;
        } else {
            let mut partition_usage = self.memory_usage.lock().unwrap();
            if granted_size > real_size {
                partition_usage.entry(consumer.partition_id()) -=
                    granted_size - real_size;
            } else {
                // TODO: this would have caused OOM already if size estimation ahead is much smaller than
                // that of actual allocation
                partition_usage.entry(consumer.partition_id()) +=
                    real_size - granted_size;
            }
        }
    }

    fn release_memory(&self, release_size: usize, partition_id: usize) {
        let mut partition_usage = self.memory_usage.lock().unwrap();
        let current_mem = partition_usage[partition_id].unwrap_or(0);
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
            partition_usage.entry(partition_id) -= to_free;
            if partition_usage[partition_id].unwrap() == 0 {
                partition_usage.remove(&partition_id);
            }
        }
        self.condvar.notify_all();
    }

    fn release_all(&self, partition_id: usize) -> usize {
        let mut partition_usage = self.memory_usage.lock().unwrap();
        let current_mem = partition_usage[partition_id].unwrap_or(0);
        if current_mem == 0 {
            return 0;
        }
        partition_usage.remove(&partition_id);
        self.condvar.notify_all();
        return current_mem;
    }
}
