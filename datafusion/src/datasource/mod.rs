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

//! DataFusion data sources

pub mod csv;
pub mod datasource;
pub mod empty;
pub mod json;
pub mod local;
pub mod memory;
pub mod object_store;
pub mod parquet;
pub mod parquet_desc;

pub use self::csv::{CsvFile, CsvReadOptions};
pub use self::datasource::{TableProvider, TableType};
pub use self::memory::MemTable;

use crate::arrow::datatypes::{Schema, SchemaRef};
use crate::datasource::datasource::{ColumnStatistics, Statistics};
use crate::datasource::object_store::ObjectStore;
use crate::error::{DataFusionError, Result};
use crate::scalar::ScalarValue;
use std::sync::Arc;

/// Source for table input data
pub(crate) enum Source<R = Box<dyn std::io::Read + Send + Sync + 'static>> {
    /// Path to a single file or a directory containing one of more files
    Path(String),

    /// Read data from a reader
    Reader(std::sync::Mutex<Option<R>>),
}

#[derive(Debug, Clone)]
pub struct PartitionedFile {
    pub file_path: String,
    pub schema: Schema,
    pub statistics: Statistics,
    pub partition_value: Option<ScalarValue>,
    pub partition_schema: Option<Schema>,
    // We may include row group range here for a more fine-grained parallel execution
}

impl std::fmt::Display for PartitionedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "PartitionedFile(file_path: {}, schema: {}, statistics: {:?},\
         partition_value: {:?}, partition_schema: {:?})",
            self.file_path,
            self.schema,
            self.statistics,
            self.partition_value,
            self.partition_schema
        )
    }
}

#[derive(Debug, Clone)]
pub struct FilePartition {
    pub index: usize,
    pub files: Vec<PartitionedFile>,
}

impl std::fmt::Display for FilePartition {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let files: Vec<String> = self.files.iter().map(|f| format!("{}", f)).collect();
        write!(
            f,
            "FilePartition[{}], files: {}",
            self.index,
            files.join(", ")
        )
    }
}

#[derive(Debug, Clone)]
pub struct SourceRootDescriptor {
    pub partition_files: Vec<PartitionedFile>,
    pub schema: SchemaRef,
}

pub trait SourceRootDescBuilder {
    fn get_source_desc(
        root_path: &str,
        object_store: Arc<dyn ObjectStore>,
        ext: &str,
    ) -> Result<SourceRootDescriptor> {
        let filenames = object_store.list_all_files(root_path, ext)?;
        if filenames.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "No file (with .{} extension) found at path {}",
                ext, root_path
            )));
        }

        // build a list of Parquet partitions with statistics and gather all unique schemas
        // used in this data set
        let mut schemas: Vec<Schema> = vec![];

        let partitioned_files = filenames
            .iter()
            .map(|file_path| {
                let pf = Self::get_file_meta(file_path, object_store)?;
                let schema = pf.schema.clone();
                if schemas.is_empty() {
                    schemas.push(schema);
                } else if schema != schemas[0] {
                    // we currently get the schema information from the first file rather than do
                    // schema merging and this is a limitation.
                    // See https://issues.apache.org/jira/browse/ARROW-11017
                    return Err(DataFusionError::Plan(format!(
                        "The file {} have different schema from the first file and DataFusion does \
                        not yet support schema merging",
                        file_path
                    )));
                }
                Ok(pf)
            }).collect::<Result<Vec<PartitionedFile>>>();

        Ok(SourceRootDescriptor {
            partition_files: partitioned_files?,
            schema: Arc::new(schemas.pop().unwrap()),
        })
    }

    fn get_file_meta(
        file_path: &str,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<PartitionedFile>;
}

pub fn get_statistics_with_limit(
    source_desc: &SourceRootDescriptor,
    limit: Option<usize>,
) -> (Vec<PartitionedFile>, Statistics) {
    let mut all_files = source_desc.partition_files.clone();
    let schema = source_desc.schema.clone();

    let mut total_byte_size = 0;
    let mut null_counts = vec![0; schema.fields().len()];

    let mut num_rows = 0;
    let mut num_files = 0;
    for file in all_files {
        num_files += 1;
        let file_stats = file.statistics;
        num_rows += file_stats.num_rows.unwrap_or(0);
        total_byte_size += file_stats.total_byte_size.unwrap_or(0);
        if let Some(vec) = file_stats.column_statistics {
            for (i, cs) in vec.iter().enumerate() {
                null_counts[i] += cs.null_count.unwrap_or(0);
            }
        }
        if num_rows > limit.unwrap_or(usize::MAX) {
            break;
        }
    }
    all_files.truncate(num_files);

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
    (all_files, statistics)
}
