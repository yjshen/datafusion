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

use crate::datasource::datasource::{ColumnStatistics, Statistics};
use crate::datasource::object_store::{ObjectReader, ObjectStore};
use crate::datasource::{
    get_statistics_with_limit, PartitionedFile, SourceRootDescBuilder,
    SourceRootDescriptor,
};
use crate::error::Result;
use crate::execution::context::ExecutionContext;
use arrow::datatypes::SchemaRef;

use crate::parquet::file::reader::Length;
use parquet::arrow::ArrowReader;
use parquet::arrow::ParquetFileArrowReader;
use parquet::file::reader::ChunkReader;
use parquet::file::serialized_reader::SerializedFileReader;
use std::io::Read;
use std::sync::Arc;

#[derive(Debug)]
pub struct ParquetRootDesc {
    pub object_store: Arc<dyn ObjectStore>,
    pub descriptor: SourceRootDescriptor,
}

pub struct X {
    or: Arc<dyn ObjectReader>,
}

impl X {
    pub fn new(or: Arc<dyn ObjectReader>) -> Self {
        Self { or }
    }
}

impl Read for X {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        todo!()
    }
}

impl ChunkReader for X {
    type T = Self;

    fn get_read(&self, start: u64, length: usize) -> parquet::errors::Result<Self::T> {
        todo!()
    }
}

impl Length for X {
    fn len(&self) -> u64 {
        todo!()
    }
}

impl ParquetRootDesc {
    pub fn new(root_path: &str) -> Result<Self> {
        let object_store = ExecutionContext::get()
            .state
            .lock()
            .unwrap()
            .protocol_registry
            .store_for_path(root_path);
        let root_desc = Self::get_source_desc(root_path, object_store.clone(), "parquet");
        Ok(Self {
            object_store,
            descriptor: root_desc?,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.descriptor.schema.clone()
    }

    pub fn statistics(&self) -> Statistics {
        get_statistics_with_limit(&self.descriptor, None).1
    }
}

impl SourceRootDescBuilder for ParquetRootDesc {
    fn get_file_meta(
        file_path: &str,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<PartitionedFile> {
        let reader = object_store.get_reader(file_path)?;
        let file_reader = Arc::new(SerializedFileReader::new(X::new(reader))?);
        let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
        let file_path = file_path.to_string();
        let schema = arrow_reader.get_schema()?;
        let num_fields = schema.fields().len();
        let meta_data = arrow_reader.get_metadata();

        let mut num_rows = 0;
        let mut total_byte_size = 0;
        let mut null_counts = vec![0; num_fields];

        for row_group_meta in meta_data.row_groups() {
            num_rows += row_group_meta.num_rows();
            total_byte_size += row_group_meta.total_byte_size();

            let columns_null_counts = row_group_meta
                .columns()
                .iter()
                .flat_map(|c| c.statistics().map(|stats| stats.null_count()));

            for (i, cnt) in columns_null_counts.enumerate() {
                null_counts[i] += cnt
            }
        }

        let column_stats = null_counts
            .iter()
            .map(|null_count| ColumnStatistics {
                null_count: Some(*null_count as usize),
                max_value: None,
                min_value: None,
                distinct_count: None,
            })
            .collect();

        let statistics = Statistics {
            num_rows: Some(num_rows as usize),
            total_byte_size: Some(total_byte_size as usize),
            column_statistics: Some(column_stats),
        };

        Ok(PartitionedFile {
            file_path: file_path.clone(),
            schema,
            statistics,
            partition_value: None,
            partition_schema: None,
        })
    }
}
