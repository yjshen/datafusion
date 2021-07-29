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
use crate::error::{DataFusionError, Result};
use crate::scalar::ScalarValue;
use arrow::datatypes::{Schema, SchemaRef};

use parquet::arrow::ArrowReader;
use parquet::arrow::ParquetFileArrowReader;
use parquet::file::reader::ChunkReader;
use parquet::file::serialized_reader::SerializedFileReader;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PartitionedFile {
    pub file_path: String,
    pub schema: Schema,
    pub statistics: Statistics,
    pub partition_value: Option<ScalarValue>,
    pub partition_schema: Option<Schema>,
    // We may include row group range here for a more fine-grained parallel execution
}

#[derive(Debug, Clone)]
pub struct FilePartition {
    pub index: usize,
    pub files: Vec<PartitionedFile>,
}

#[derive(Debug, Clone)]
pub struct SourceDescriptor {
    pub partition_files: Vec<PartitionedFile>,
    pub schema: SchemaRef,
}

pub trait DataSource2: Send + Sync {
    fn list_partitions(&self, max_concurrency: usize) -> Result<Arc<FilePartition>>;

    fn schema(&self) -> Result<Arc<Schema>>;

    fn get_read_for_file(
        &self,
        partitioned_file: PartitionedFile,
    ) -> Result<dyn ChunkReader>;

    fn statistics(&self) -> &Statistics;
}

pub trait SourceDescBuilder {
    fn get_source_desc(root_path: &str) -> Result<SourceDescriptor> {
        let filenames = Self::get_all_files(root_path)?;
        if filenames.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "No Parquet files (with .parquet extension) found at path {}",
                root_path
            )));
        }

        // build a list of Parquet partitions with statistics and gather all unique schemas
        // used in this data set
        let mut schemas: Vec<Schema> = vec![];

        let partitioned_files = filenames
            .iter()
            .map(|file_path| {
                let pf = Self::get_file_meta(file_path)?;
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

        Ok(SourceDescriptor {
            partition_files: partitioned_files?,
            schema: Arc::new(schemas.pop().unwrap()),
        })
    }

    fn get_all_files(root_path: &str) -> Result<Vec<String>>;

    fn get_file_meta(file_path: &str) -> Result<PartitionedFile>;

    fn reader_for_file_meta(file_path: &str) -> Result<dyn ChunkReader>;
}

pub trait ParquetSourceDescBuilder: SourceDescBuilder {
    fn get_file_meta(file_path: &str) -> Result<PartitionedFile> {
        let chunk_reader = Self::reader_for_file_meta(file_path)?;
        let file_reader = Arc::new(SerializedFileReader::new(chunk_reader)?);
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
