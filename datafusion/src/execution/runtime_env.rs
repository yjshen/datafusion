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

use crate::error::Result;
use crate::execution::disk_manager::DiskManager;
use crate::execution::memory_management::MemoryManager;
use std::sync::Arc;

// Employ lazy static temporarily for RuntimeEnv, to avoid plumbing it through
// all `async fn execute(&self, partition: usize, runtime: Arc<RuntimeEnv>)`
lazy_static! {
    pub static ref RUNTIME_ENV: Arc<RuntimeEnv> = {
        let config = RuntimeConfig::new();
        Arc::new(RuntimeEnv::new(config).unwrap())
    };
}

#[derive(Clone)]
pub struct RuntimeEnv {
    pub config: RuntimeConfig,
    /// Runtime memory management
    pub memory_manager: Arc<MemoryManager>,
    /// Manage temporary files during query execution
    pub disk_manager: Arc<DiskManager>,
}

impl RuntimeEnv {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let memory_manager = Arc::new(MemoryManager::new(config.max_memory));
        let disk_manager = Arc::new(DiskManager::new(&config.local_dirs)?);
        Ok(Self {
            config,
            memory_manager,
            disk_manager,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }
}

struct RuntimeConfig {
    /// Default batch size when creating new batches
    pub batch_size: usize,
    /// Max execution memory allowed for DataFusion
    pub max_memory: usize,
    /// Local dirs to store temporary files during execution
    pub local_dirs: Vec<String>,
}

impl RuntimeConfig {
    pub fn new() -> Self {
        Default::default()
    }

    /// Customize batch size
    pub fn with_batch_size(mut self, n: usize) -> Self {
        // batch size must be greater than zero
        assert!(n > 0);
        self.batch_size = n;
        self
    }

    /// Customize exec size
    pub fn with_max_execution_memory(mut self, max_memory: usize) -> Self {
        assert!(max_memory > 0);
        self.max_memory = max_memory;
        self
    }

    /// Customize exec size
    pub fn with_local_dirs(mut self, local_dirs: Vec<String>) -> Self {
        assert!(local_dirs.len() > 0);
        self.local_dirs = local_dirs;
        self
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().to_str().unwrap().to_string();
        std::mem::forget(tmp_dir);

        Self {
            batch_size: 8192,
            max_memory: usize::MAX,
            local_dirs: vec![path],
        }
    }
}
