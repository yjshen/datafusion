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

use crate::datasource::local::LocalFSHandler;
use crate::error::Result;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use parquet::file::reader::ChunkReader;

pub trait ProtocolHandler: Sync + Send {
    /// Returns the protocol handler as [`Any`](std::any::Any)
    /// so that it can be downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    fn list_all_files(&self, root_path: &str, ext: &str) -> Result<Vec<String>>;

    fn get_reader(&self, file_path: &str) -> Result<Arc<dyn ChunkReader>>;

    fn handler_name(&self) -> String;
}

impl Debug for dyn ProtocolHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.handler_name())
    }
}

static LOCAL_SCHEME: &str = "file";

pub struct ProtocolRegistry {
    pub protocol_handlers: RwLock<HashMap<String, Arc<dyn ProtocolHandler>>>,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        let mut map: HashMap<String, Arc<dyn ProtocolHandler>> = HashMap::new();
        map.insert(LOCAL_SCHEME.to_string(), Arc::new(LocalFSHandler));

        Self {
            protocol_handlers: RwLock::new(map),
        }
    }

    /// Adds a new handler to this registry.
    /// If a handler of the same prefix existed before, it is replaced in the registry and returned.
    pub fn register_handler(
        &self,
        scheme: &str,
        handler: Arc<dyn ProtocolHandler>,
    ) -> Option<Arc<dyn ProtocolHandler>> {
        let mut handlers = self.protocol_handlers.write().unwrap();
        handlers.insert(scheme.to_string(), handler)
    }

    pub fn handler(&self, scheme: &str) -> Option<Arc<dyn ProtocolHandler>> {
        let handlers = self.protocol_handlers.read().unwrap();
        handlers.get(scheme).cloned()
    }

    pub fn handler_for_path(&self, path: &str) -> Arc<dyn ProtocolHandler> {
        if let Some((scheme, _)) = path.split_once(':') {
            let handlers = self.protocol_handlers.read().unwrap();
            if let Some(handler) = handlers.get(&*scheme.to_lowercase()) {
                return handler.clone();
            }
        }
        self.protocol_handlers
            .read()
            .unwrap()
            .get(LOCAL_SCHEME)
            .unwrap()
            .clone()
    }
}
