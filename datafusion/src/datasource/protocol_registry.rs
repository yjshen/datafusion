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

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::datasource2::DataSource2;
use crate::error::{DataFusionError, Result};
use crate::parquet::file::reader::ChunkReader;
use std::any::Any;
use std::fs::File;

pub trait ProtocolHandler: Sync + Send {
    /// Returns the protocol handler as [`Any`](std::any::Any)
    /// so that it can be downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    fn list_all_files(&self, root_path: &str, ext: &str) -> Result<Vec<String>>;

    fn get_reader(&self, file_path: &str) -> Result<dyn ChunkReader>;
}

pub struct LocalFSHandler;

impl ProtocolHander for LocalFSHander {
    fn as_any(&self) -> &dyn Any {
        return self;
    }

    fn list_all_files(&self, root_path: &str, ext: &str) -> Result<Vec<String>> {
        let mut filenames: Vec<String> = Vec::new();
        crate::datasource::local::list_all_files(root_path, &mut filenames, ext);
        Ok(filenames)
    }

    fn get_reader(&self, file_path: &str) -> Result<R> {
        Ok(File::open(file_path)?)
    }
}

pub struct ProtocolRegistry {
    pub protocol_handlers: RwLock<HashMap<String, Arc<dyn ProtocolHandler>>>,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        Self {
            protocol_handlers: RwLock::new(HashMap::new()),
        }
    }

    /// Adds a new handler to this registry.
    /// If a handler of the same prefix existed before, it is replaced in the registry and returned.
    pub fn register_handler(
        &self,
        prefix: &str,
        handler: Arc<dyn ProtocolHander>,
    ) -> Option<Arc<dyn ProtocolHander>> {
        let mut handlers = self.protocol_handlers.write().unwrap();
        handlers.insert(prefix.to_string(), handler)
    }

    pub fn handler(&self, prefix: &str) -> Option<Arc<dyn ProtocolHander>> {
        let handlers = self.protocol_handlers.read().unwrap();
        handlers.get(prefix).cloned()
    }
}
