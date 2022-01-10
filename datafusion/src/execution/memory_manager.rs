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

use crate::error::Result;
use async_trait::async_trait;
use hashbrown::HashMap;
use log::info;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

static mut CONSUMER_ID: AtomicUsize = AtomicUsize::new(0);

fn next_id() -> usize {
    unsafe { CONSUMER_ID.fetch_add(1, Ordering::SeqCst) }
}

/// Type of the memory consumer
pub enum ConsumerType {
    /// consumers that can grow its memory usage by requesting more from the memory manager or
    /// shrinks its memory usage when we can no more assign available memory to it.
    /// Examples are spillable sorter, spillable hashmap, etc.
    Requesting,
    /// consumers that are not spillable, counting in for only tracking purpose.
    Tracking,
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
/// A memory consumer that either takes up memory (of type `ConsumerType::Tracking`)
/// or grows/shrinks memory usage based on available memory (of type `ConsumerType::Requesting`).
pub trait MemoryConsumer: Send + Sync + Debug {
    /// Display name of the consumer
    fn name(&self) -> String;

    /// Unique id of the consumer
    fn id(&self) -> &MemoryConsumerId;

    /// Ptr to MemoryManager
    fn memory_manager(&self) -> Arc<MemoryManager>;

    /// Partition that the consumer belongs to
    fn partition_id(&self) -> usize {
        self.id().partition_id
    }

    /// Type of the consumer
    fn type_(&self) -> &ConsumerType;

    /// Grow memory by `required` to buffer more data in memory,
    /// this may trigger spill before grow when the memory threshold is
    /// reached for this consumer.
    async fn try_grow(&self, required: usize) -> Result<()> {
        let current = self.mem_used();
        let can_grow = self
            .memory_manager()
            .can_grow(required, current, self.id())
            .await;
        if !can_grow {
            info!(
                "Failed to grow memory of {} from {}, spilling...",
                human_readable_size(required),
                self.id()
            );
            self.spill().await?;
        }
        Ok(())
    }

    /// Spill in-memory buffers to disk, free memory
    async fn spill(&self) -> Result<()>;

    /// Current memory used by this consumer
    fn mem_used(&self) -> usize;
}

impl Display for dyn MemoryConsumer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}]: {}",
            self.name(),
            self.id(),
            human_readable_size(self.mem_used())
        )
    }
}

/*
The memory management architecture is the following:

1. User designates max execution memory by setting RuntimeConfig.max_memory and RuntimeConfig.memory_fraction (float64 between 0..1).
   The actual max memory DataFusion could use `pool_size =  max_memory * memory_fraction`.
2. The entities that take up memory during its execution are called 'Memory Consumers'. Operators or others are encouraged to
   register themselves to the memory manager and report its usage through `mem_used()`.
3. There are two kinds of consumers:
   - 'Requesting' consumers that would acquire memory during its execution and release memory through `spill` if no more memory is available.
   - 'Tracking' consumers that exist for reporting purposes to provide a more accurate memory usage estimation for memory consumers.
4. Requesting and tracking consumers share the pool. Each controlling consumer could acquire a maximum of
   (pool_size - all_tracking_used) / active_num_controlling_consumers.

            Memory Space for the DataFusion Lib / Process of `pool_size`
   ┌──────────────────────────────────────────────z─────────────────────────────┐
   │                                              z                             │
   │                                              z                             │
   │               Requesting                     z          Tracking           │
   │            Memory Consumers                  z       Memory Consumers      │
   │                                              z                             │
   │                                              z                             │
   └──────────────────────────────────────────────z─────────────────────────────┘
*/

/// Manage memory usage during physical plan execution
pub struct MemoryManager {
    requesters: Arc<Mutex<HashMap<MemoryConsumerId, Weak<dyn MemoryConsumer>>>>,
    trackers: Arc<Mutex<HashMap<MemoryConsumerId, Weak<dyn MemoryConsumer>>>>,
    pool_size: usize,
}

impl MemoryManager {
    /// Create new memory manager based on max available pool_size
    pub fn new(pool_size: usize) -> Self {
        info!(
            "Creating memory manager with initial size {}",
            human_readable_size(pool_size)
        );
        Self {
            requesters: Arc::new(Mutex::new(HashMap::new())),
            trackers: Arc::new(Mutex::new(HashMap::new())),
            pool_size,
        }
    }

    fn get_tracker_total(&self) -> usize {
        let trackers = self.trackers.lock().unwrap();
        if trackers.len() > 0 {
            trackers.values().fold(0usize, |acc, y| match y.upgrade() {
                None => acc,
                Some(t) => acc + t.mem_used(),
            })
        } else {
            0
        }
    }

    /// Register a new memory consumer for memory usage tracking
    pub(crate) fn register_consumer(
        self: &Arc<Self>,
        consumer: &Arc<dyn MemoryConsumer>,
    ) {
        let id = consumer.id().clone();
        match consumer.type_() {
            ConsumerType::Requesting => {
                let mut requesters = self.requesters.lock().unwrap();
                requesters.insert(id, Arc::downgrade(consumer));
            }
            ConsumerType::Tracking => {
                let mut trackers = self.trackers.lock().unwrap();
                trackers.insert(id, Arc::downgrade(consumer));
            }
        }
    }

    /// Grow memory attempt from a consumer, return if we could grant that much to it
    async fn can_grow(
        self: &Arc<Self>,
        required: usize,
        current: usize,
        consumer_id: &MemoryConsumerId,
    ) -> bool {
        let tracker_total = self.get_tracker_total();
        let max_per_op = {
            let total_available = self.pool_size - tracker_total;
            let ops = self.requesters.lock().unwrap().len();
            (total_available / ops) as usize
        };
        let granted = required + current < max_per_op;
        info!(
            "trying to acquire {} whiling holding {} from {}, got: {}",
            human_readable_size(required),
            human_readable_size(current),
            consumer_id,
            granted,
        );
        granted
    }

    /// Drop a memory consumer from memory usage tracking
    pub(crate) fn drop_consumer(self: &Arc<Self>, id: &MemoryConsumerId) {
        // find in requesters first
        {
            let mut requesters = self.requesters.lock().unwrap();
            if requesters.remove(id).is_some() {
                return;
            }
        }
        let mut trackers = self.trackers.lock().unwrap();
        trackers.remove(id);
    }
}

impl Display for MemoryManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let requesters =
            self.requesters
                .lock()
                .unwrap()
                .values()
                .fold(vec![], |mut acc, consumer| match consumer.upgrade() {
                    None => acc,
                    Some(c) => {
                        acc.push(format!("{}", c));
                        acc
                    }
                });
        let tracker_mem = self.get_tracker_total();
        write!(f,
               "MemoryManager usage statistics: total {}, tracker used {}, total {} requesters detail: \n {},",
                human_readable_size(self.pool_size),
                human_readable_size(tracker_mem),
                &requesters.len(),
               requesters.join("\n"))
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
