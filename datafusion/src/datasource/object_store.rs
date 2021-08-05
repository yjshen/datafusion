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

use crate::datasource::local::LocalFSHandler;
use crate::error::Result;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Read;
use std::sync::{Arc, RwLock};

pub trait ObjectReader {
    fn get_iter(&self) -> Box<dyn Iterator<Item = u8>>;
    fn len1(& self) -> u64;
}

/// A ObjectStore abstracts access to an underlying file/object storage.
/// It maps strings (e.g. URLs, filesystem paths, etc) to sources of bytes
pub trait ObjectStore: Sync + Send + Debug {
    /// Returns the protocol handler as [`Any`](std::any::Any)
    /// so that it can be downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    fn list_all_files(&self, path: &str, ext: &str) -> Result<Vec<String>>;

    fn get_reader(&self, file_path: &str) -> Result<Arc<dyn ObjectReader>>;

    fn handler_name(&self) -> String;
}

static LOCAL_SCHEME: &str = "file";

pub struct ObjectStoreRegistry {
    pub object_stores: RwLock<HashMap<String, Arc<dyn ObjectStore>>>,
}

impl ObjectStoreRegistry {
    pub fn new() -> Self {
        let mut map: HashMap<String, Arc<dyn ObjectStore>> = HashMap::new();
        map.insert(LOCAL_SCHEME.to_string(), Arc::new(LocalFSHandler));

        Self {
            object_stores: RwLock::new(map),
        }
    }

    /// Adds a new store to this registry.
    /// If a store of the same prefix existed before, it is replaced in the registry and returned.
    pub fn register_store(
        &self,
        scheme: &str,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        let mut stores = self.object_stores.write().unwrap();
        stores.insert(scheme.to_string(), store)
    }

    pub fn get(&self, scheme: &str) -> Option<Arc<dyn ObjectStore>> {
        let stores = self.object_stores.read().unwrap();
        stores.get(scheme).cloned()
    }

    /// Get a suitable store for the path based on it's scheme. For example:
    /// path with prefix file:/// or no prefix will return the default LocalFS store,
    /// path with prefix s3:/// will return the S3 store if it's registered,
    /// and will always return LocalFS store when a prefix is not registered in the path.
    pub fn store_for_path(&self, path: &str) -> Arc<dyn ObjectStore> {
        if let Some((scheme, _)) = path.split_once(':') {
            let stores = self.object_stores.read().unwrap();
            if let Some(handler) = stores.get(&*scheme.to_lowercase()) {
                return handler.clone();
            }
        }
        self.object_stores
            .read()
            .unwrap()
            .get(LOCAL_SCHEME)
            .unwrap()
            .clone()
    }
}
