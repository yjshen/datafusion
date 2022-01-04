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

//! Defines the External-Sort plan

use crate::error::{DataFusionError, Result};
use crate::execution::memory_management::{
    MemoryConsumer, MemoryConsumerId, MemoryManager,
};
use crate::execution::runtime_env::RuntimeEnv;
use crate::execution::runtime_env::RUNTIME_ENV;
use crate::physical_plan::common::{
    batch_memory_size, IPCWriterWrapper, SizedRecordBatchStream,
};
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::sorts::in_mem_sort::InMemSortStream;
use crate::physical_plan::sorts::sort::sort_batch;
use crate::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeStream;
use crate::physical_plan::sorts::SpillableStream;
use crate::physical_plan::stream::RecordBatchReceiverStream;
use crate::physical_plan::{
    DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    SendableRecordBatchStream, Statistics,
};
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::io::ipc::read::{read_file_metadata, FileReader};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::lock::Mutex;
use futures::StreamExt;
use log::{error, info};
use std::any::Any;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver as TKReceiver, Sender as TKSender};
use tokio::task;

struct ExternalSorter {
    id: MemoryConsumerId,
    schema: SchemaRef,
    in_mem_batches: Mutex<Vec<RecordBatch>>,
    spills: Mutex<Vec<String>>,
    /// Sort expressions
    expr: Vec<PhysicalSortExpr>,
    runtime: Arc<RuntimeEnv>,
    metrics: ExecutionPlanMetricsSet,
    used: AtomicIsize,
    spilled_bytes: AtomicUsize,
    spilled_count: AtomicUsize,
    insert_finished: AtomicBool,
}

impl ExternalSorter {
    pub fn new(
        partition_id: usize,
        schema: SchemaRef,
        expr: Vec<PhysicalSortExpr>,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        Self {
            id: MemoryConsumerId::new(partition_id),
            schema,
            in_mem_batches: Mutex::new(vec![]),
            spills: Mutex::new(vec![]),
            expr,
            runtime,
            metrics: ExecutionPlanMetricsSet::new(),
            used: AtomicIsize::new(0),
            spilled_bytes: AtomicUsize::new(0),
            spilled_count: AtomicUsize::new(0),
            insert_finished: AtomicBool::new(false),
        }
    }

    pub(crate) fn finish_insert(&self) {
        self.insert_finished.store(true, Ordering::SeqCst);
    }

    async fn spill_while_inserting(&self) -> Result<usize> {
        info!(
            "{} spilling sort data of {} to disk while inserting ({} time(s) so far)",
            self.str_repr(),
            self.get_used(),
            self.spilled_count()
        );

        let partition = self.partition_id();
        let mut in_mem_batches = self.in_mem_batches.lock().await;
        // we could always get a chance to free some memory as long as we are holding some
        if in_mem_batches.len() == 0 {
            return Ok(0);
        }

        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let path = self.runtime.disk_manager.create_tmp_file()?;
        let stream = in_mem_merge_sort(
            &mut *in_mem_batches,
            self.schema.clone(),
            &*self.expr,
            self.runtime.batch_size(),
            baseline_metrics,
        )
        .await;

        let total_size = spill(&mut stream?, path.clone(), self.schema.clone()).await?;

        let mut spills = self.spills.lock().await;
        self.spilled_count.fetch_add(1, Ordering::SeqCst);
        self.spilled_bytes.fetch_add(total_size, Ordering::SeqCst);
        spills.push(path);
        Ok(total_size)
    }

    async fn insert_batch(&self, input: RecordBatch) -> Result<()> {
        let size = batch_memory_size(&input);
        self.allocate(size).await?;
        // sort each batch as it's inserted, more probably to be cache-resident
        let sorted_batch = sort_batch(input, self.schema.clone(), &*self.expr)?;
        let mut in_mem_batches = self.in_mem_batches.lock().await;
        in_mem_batches.push(sorted_batch);
        Ok(())
    }

    /// MergeSort in mem batches as well as spills into total order with `SortPreservingMergeStream`(SPMS).
    /// Always put in mem batch based stream to idx 0 in SPMS so that we could spill
    /// the stream when `spill()` is called on us.
    async fn sort(&self) -> Result<SendableRecordBatchStream> {
        let partition = self.partition_id();
        let mut in_mem_batches = self.in_mem_batches.lock().await;
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let mut streams: Vec<SpillableStream> = vec![];
        let in_mem_stream = in_mem_merge_sort(
            &mut *in_mem_batches,
            self.schema.clone(),
            &self.expr,
            self.runtime.batch_size(),
            baseline_metrics,
        )
        .await?;
        streams.push(SpillableStream::new_spillable(in_mem_stream));

        let mut spills = self.spills.lock().await;

        for spill in spills.drain(..) {
            let stream = read_spill_as_stream(spill, self.schema.clone()).await?;
            streams.push(SpillableStream::new_unspillable(stream));
        }
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        Ok(Box::pin(
            SortPreservingMergeStream::new_from_stream(
                streams,
                self.schema.clone(),
                &self.expr,
                self.runtime.batch_size(),
                baseline_metrics,
                partition,
                self.runtime.clone(),
            )
            .await,
        ))
    }
}

impl Debug for ExternalSorter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalSorter")
            .field("id", &self.id())
            .field("memory_used", &self.get_used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spilled_count", &self.spilled_count())
            .finish()
    }
}

#[async_trait]
impl MemoryConsumer for ExternalSorter {
    fn name(&self) -> String {
        "ExternalSorter".to_owned()
    }

    fn id(&self) -> &MemoryConsumerId {
        &self.id
    }

    fn memory_manager(&self) -> Arc<MemoryManager> {
        self.runtime.memory_manager.clone()
    }

    async fn spill_inner(
        &self,
        _size: usize,
        _trigger: &MemoryConsumerId,
    ) -> Result<usize> {
        if !self.insert_finished.load(Ordering::SeqCst) {
            let total_size = self.spill_while_inserting().await;
            total_size
        } else {
            Ok(0)
        }
    }

    fn get_used(&self) -> isize {
        self.used.load(Ordering::SeqCst)
    }

    fn update_used(&self, delta: isize) {
        self.used.fetch_add(delta, Ordering::SeqCst);
    }

    fn spilled_bytes(&self) -> usize {
        self.spilled_bytes.load(Ordering::SeqCst)
    }

    fn spilled_bytes_add(&self, add: usize) {
        self.spilled_bytes.fetch_add(add, Ordering::SeqCst);
    }

    fn spilled_count(&self) -> usize {
        self.spilled_count.load(Ordering::SeqCst)
    }

    fn spilled_count_increment(&self) {
        self.spilled_count.fetch_add(1, Ordering::SeqCst);
    }
}

/// consume the `sorted_bathes` and do in_mem_sort
async fn in_mem_merge_sort(
    sorted_bathes: &mut Vec<RecordBatch>,
    schema: SchemaRef,
    expressions: &[PhysicalSortExpr],
    target_batch_size: usize,
    baseline_metrics: BaselineMetrics,
) -> Result<SendableRecordBatchStream> {
    if sorted_bathes.len() == 1 {
        Ok(Box::pin(SizedRecordBatchStream::new(
            schema,
            vec![Arc::new(sorted_bathes.pop().unwrap())],
        )))
    } else {
        let new = sorted_bathes.drain(..).collect();
        assert_eq!(sorted_bathes.len(), 0);
        Ok(Box::pin(InMemSortStream::new(
            new,
            schema,
            expressions,
            target_batch_size,
            baseline_metrics,
        )?))
    }
}

async fn spill(
    in_mem_stream: &mut SendableRecordBatchStream,
    path: String,
    schema: SchemaRef,
) -> Result<usize> {
    let (sender, receiver): (
        TKSender<ArrowResult<RecordBatch>>,
        TKReceiver<ArrowResult<RecordBatch>>,
    ) = tokio::sync::mpsc::channel(2);
    while let Some(item) = in_mem_stream.next().await {
        sender.send(item).await.ok();
    }
    let path_clone = path.clone();
    let res =
        task::spawn_blocking(move || write_sorted(receiver, path_clone, schema)).await;
    match res {
        Ok(r) => r,
        Err(e) => Err(DataFusionError::Execution(format!(
            "Error occurred while spilling {}",
            e
        ))),
    }
}

async fn read_spill_as_stream(
    path: String,
    schema: SchemaRef,
) -> Result<SendableRecordBatchStream> {
    let (sender, receiver): (
        TKSender<ArrowResult<RecordBatch>>,
        TKReceiver<ArrowResult<RecordBatch>>,
    ) = tokio::sync::mpsc::channel(2);
    let path_clone = path.clone();
    let join_handle = task::spawn_blocking(move || {
        if let Err(e) = read_spill(sender, path_clone) {
            error!("Failure while reading spill file: {}. Error: {}", path, e);
        }
    });
    Ok(RecordBatchReceiverStream::create(
        &schema,
        receiver,
        join_handle,
    ))
}

pub(crate) async fn convert_stream_disk_based(
    in_mem_stream: &mut SendableRecordBatchStream,
    path: String,
    schema: SchemaRef,
) -> Result<(SendableRecordBatchStream, usize)> {
    let size = spill(in_mem_stream, path.clone(), schema.clone()).await?;
    read_spill_as_stream(path.clone(), schema.clone())
        .await
        .map(|s| (s, size))
}

fn write_sorted(
    mut receiver: TKReceiver<ArrowResult<RecordBatch>>,
    path: String,
    schema: SchemaRef,
) -> Result<usize> {
    let mut writer = IPCWriterWrapper::new(path.as_ref(), schema.as_ref())?;
    while let Some(batch) = receiver.blocking_recv() {
        writer.write(&batch?)?;
    }
    writer.finish()?;
    info!(
        "Spilled {} batches of total {} rows to disk, memory released {}",
        writer.num_batches, writer.num_rows, writer.num_bytes
    );
    Ok(writer.num_bytes as usize)
}

fn read_spill(sender: TKSender<ArrowResult<RecordBatch>>, path: String) -> Result<()> {
    let mut file = BufReader::new(File::open(&path)?);
    let file_meta = read_file_metadata(&mut file)?;
    let reader = FileReader::new(&mut file, file_meta, None);
    for batch in reader {
        sender
            .blocking_send(batch)
            .map_err(|e| DataFusionError::Execution(format!("{}", e)))?;
    }
    Ok(())
}

/// Sort execution plan
#[derive(Debug)]
pub struct ExternalSortExec {
    /// Input schema
    input: Arc<dyn ExecutionPlan>,
    /// Sort expressions
    expr: Vec<PhysicalSortExpr>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Preserve partitions of input plan
    preserve_partitioning: bool,
}

impl ExternalSortExec {
    /// Create a new sort execution plan
    pub fn try_new(
        expr: Vec<PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        Ok(Self::new_with_partitioning(expr, input, false))
    }

    /// Create a new sort execution plan with the option to preserve
    /// the partitioning of the input plan
    pub fn new_with_partitioning(
        expr: Vec<PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
    ) -> Self {
        Self {
            expr,
            input,
            metrics: ExecutionPlanMetricsSet::new(),
            preserve_partitioning,
        }
    }

    /// Input schema
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Sort expressions
    pub fn expr(&self) -> &[PhysicalSortExpr] {
        &self.expr
    }
}

#[async_trait]
impl ExecutionPlan for ExternalSortExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        if self.preserve_partitioning {
            self.input.output_partitioning()
        } else {
            Partitioning::UnknownPartitioning(1)
        }
    }

    fn required_child_distribution(&self) -> Distribution {
        if self.preserve_partitioning {
            Distribution::UnspecifiedDistribution
        } else {
            Distribution::SinglePartition
        }
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(ExternalSortExec::try_new(
                self.expr.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "ExternalSortExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if !self.preserve_partitioning {
            if 0 != partition {
                return Err(DataFusionError::Internal(format!(
                    "ExternalSortExec invalid partition {}",
                    partition
                )));
            }

            // sort needs to operate on a single partition currently
            if 1 != self.input.output_partitioning().partition_count() {
                return Err(DataFusionError::Internal(
                    "SortExec requires a single input partition".to_owned(),
                ));
            }
        }

        let input = self.input.execute(partition).await?;
        external_sort(input, partition, self.expr.clone(), RUNTIME_ENV.clone()).await
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                let expr: Vec<String> = self.expr.iter().map(|e| e.to_string()).collect();
                write!(f, "SortExec: [{}]", expr.join(","))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        self.input.statistics()
    }
}

/// Sort based on `ExternalSorter`
pub async fn external_sort(
    mut input: SendableRecordBatchStream,
    partition_id: usize,
    expr: Vec<PhysicalSortExpr>,
    runtime: Arc<RuntimeEnv>,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let sorter = Arc::new(ExternalSorter::new(
        partition_id,
        schema.clone(),
        expr,
        runtime.clone(),
    ));
    runtime.register_consumer(sorter.clone()).await;

    while let Some(batch) = input.next().await {
        let batch = batch?;
        sorter.insert_batch(batch).await?;
    }

    sorter.finish_insert();
    sorter.sort().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::object_store::local::LocalFileSystem;
    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::expressions::col;
    use crate::physical_plan::memory::MemoryExec;
    use crate::physical_plan::{
        collect,
        file_format::{CsvExec, PhysicalPlanConfig},
    };
    use crate::test;
    use crate::test_util;
    use arrow::array::*;
    use arrow::compute::sort::SortOptions;
    use arrow::datatypes::*;

    #[tokio::test]
    async fn test_sort() -> Result<()> {
        let schema = test_util::aggr_test_schema();
        let partitions = 4;
        let (_, files) =
            test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;

        let csv = CsvExec::new(
            PhysicalPlanConfig {
                object_store: Arc::new(LocalFileSystem {}),
                file_schema: Arc::clone(&schema),
                file_groups: files,
                statistics: Statistics::default(),
                projection: None,
                batch_size: 1024,
                limit: None,
                table_partition_cols: vec![],
            },
            true,
            b',',
        );

        let sort_exec = Arc::new(ExternalSortExec::try_new(
            vec![
                // c1 string column
                PhysicalSortExpr {
                    expr: col("c1", &schema)?,
                    options: SortOptions::default(),
                },
                // c2 uin32 column
                PhysicalSortExpr {
                    expr: col("c2", &schema)?,
                    options: SortOptions::default(),
                },
                // c7 uin8 column
                PhysicalSortExpr {
                    expr: col("c7", &schema)?,
                    options: SortOptions::default(),
                },
            ],
            Arc::new(CoalescePartitionsExec::new(Arc::new(csv))),
        )?);

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        let c1 = columns[0]
            .as_any()
            .downcast_ref::<Utf8Array<i32>>()
            .unwrap();
        assert_eq!(c1.value(0), "a");
        assert_eq!(c1.value(c1.len() - 1), "e");

        let c2 = columns[1].as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(c2.value(0), 1);
        assert_eq!(c2.value(c2.len() - 1), 5,);

        let c7 = columns[6].as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(c7.value(0), 15);
        assert_eq!(c7.value(c7.len() - 1), 254,);

        Ok(())
    }

    #[tokio::test]
    async fn test_lex_sort_by_float() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, true),
            Field::new("b", DataType::Float64, true),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![
                    Some(f32::NAN),
                    None,
                    None,
                    Some(f32::NAN),
                    Some(1.0_f32),
                    Some(1.0_f32),
                    Some(2.0_f32),
                    Some(3.0_f32),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(200.0_f64),
                    Some(20.0_f64),
                    Some(10.0_f64),
                    Some(100.0_f64),
                    Some(f64::NAN),
                    None,
                    None,
                    Some(f64::NAN),
                ])),
            ],
        )?;

        let sort_exec = Arc::new(ExternalSortExec::try_new(
            vec![
                PhysicalSortExpr {
                    expr: col("a", &schema)?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: true,
                    },
                },
                PhysicalSortExpr {
                    expr: col("b", &schema)?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ],
            Arc::new(MemoryExec::try_new(&[vec![batch]], schema, None)?),
        )?);

        assert_eq!(DataType::Float32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::Float64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec.clone()).await?;
        // let metrics = sort_exec.metrics().unwrap();
        // assert!(metrics.elapsed_compute().unwrap() > 0);
        // assert_eq!(metrics.output_rows().unwrap(), 8);
        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        assert_eq!(DataType::Float32, *columns[0].data_type());
        assert_eq!(DataType::Float64, *columns[1].data_type());

        let a = columns[0].as_any().downcast_ref::<Float32Array>().unwrap();
        let b = columns[1].as_any().downcast_ref::<Float64Array>().unwrap();

        // convert result to strings to allow comparing to expected result containing NaN
        let result: Vec<(Option<String>, Option<String>)> = (0..result[0].num_rows())
            .map(|i| {
                let aval = if a.is_valid(i) {
                    Some(a.value(i).to_string())
                } else {
                    None
                };
                let bval = if b.is_valid(i) {
                    Some(b.value(i).to_string())
                } else {
                    None
                };
                (aval, bval)
            })
            .collect();

        let expected: Vec<(Option<String>, Option<String>)> = vec![
            (None, Some("10".to_owned())),
            (None, Some("20".to_owned())),
            (Some("NaN".to_owned()), Some("100".to_owned())),
            (Some("NaN".to_owned()), Some("200".to_owned())),
            (Some("3".to_owned()), Some("NaN".to_owned())),
            (Some("2".to_owned()), None),
            (Some("1".to_owned()), Some("NaN".to_owned())),
            (Some("1".to_owned()), None),
        ];

        assert_eq!(expected, result);

        Ok(())
    }
}
