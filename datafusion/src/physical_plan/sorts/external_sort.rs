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

use super::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use super::{RecordBatchStream, SendableRecordBatchStream, Statistics};
use crate::error::{DataFusionError, Result};
use crate::execution::disk_manager::{DiskManager, PathFile};
use crate::execution::memory_management::{
    MemoryConsumer, MemoryConsumerId, MemoryManager, PartitionMemoryManager,
};
use crate::execution::runtime_env::RuntimeEnv;
use crate::physical_plan::common::{
    batch_memory_size, IPCWriterWrapper, SizedRecordBatchStream,
};
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::memory::MemoryStream;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use crate::physical_plan::sort::sort_batch;
use crate::physical_plan::sort_preserving_merge::SortPreservingMergeStream;
use crate::physical_plan::sorts::sort::sort_batch;
use crate::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeStream;
use crate::physical_plan::{
    common, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use arrow::compute::aggregate::estimated_bytes_size;
pub use arrow::compute::sort::SortOptions;
use arrow::compute::{sort::lexsort_to_indices, take};
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, error::ArrowError};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{Future, SinkExt, Stream, StreamExt};
use log::{debug, info};
use parking_lot::Mutex;
use pin_project_lite::pin_project;
use std::any::Any;
use std::pin::Pin;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task;

struct ExternalSorter {
    id: MemoryConsumerId,
    schema: SchemaRef,
    in_mem_batches: Mutex<Vec<RecordBatch>>,
    spills: Mutex<Vec<String>>,
    used: AtomicIsize,
    spilled_bytes: AtomicUsize,
    spilled_count: AtomicUsize,
    /// Sort expressions
    expr: Vec<PhysicalSortExpr>,
    runtime: RuntimeEnv,
}

impl ExternalSorter {
    pub fn new(
        partition_id: usize,
        schema: SchemaRef,
        expr: Vec<PhysicalSortExpr>,
        runtime: RuntimeEnv,
    ) -> Self {
        Self {
            id: MemoryConsumerId::new(partition_id),
            schema,
            in_mem_batches: Mutex::new(vec![]),
            spills: Mutex::new(vec![]),
            used: AtomicIsize::new(0),
            spilled_bytes: AtomicUsize::new(0),
            spilled_count: AtomicUsize::new(0),
            expr,
            runtime,
        }
    }

    fn insert_batch(
        &mut self,
        input: RecordBatch,
        schema: SchemaRef,
        expr: &[PhysicalSortExpr],
    ) -> Result<()> {
        let size = batch_memory_size(&input);
        self.allocate(size)?;
        // sort each batch as it's inserted, more probably to be cache-resident
        let sorted_batch = sort_batch(input, schema, expr)?;
        let mut in_mem_batches = self.in_mem_batches.lock();
        in_mem_batches.push(sorted_batch);
    }

    fn sort(&self) {}
}

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

    async fn spill(&self, _size: usize, _trigger: &dyn MemoryConsumer) -> Result<usize> {
        let in_mem_batches = self.in_mem_batches.lock();

        // we could always get a chance to free some memory as long as we are holding some
        if in_mem_batches.len() == 0 {
            return Ok(0);
        }

        info!(
            "{} spilling sort data of {} to disk ({} time(s) so far)",
            self.str_repr(),
            self.get_used(),
            self.spilled_count()
        );

        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let total_size = in_mem_batches.iter().map(|b| batch_memory_size(b)).sum();
        let path = self.disk_manager.create_tmp_file()?;
        let stream = merge_sort(
            *in_mem_batches,
            self.schema.clone(),
            &*self.expr,
            self.runtime.batch_size(),
            baseline_metrics,
        )
        .await;

        spill(stream, path, self.schema.clone())?;

        {
            let mut spills = self.spills.lock();
            self.spilled_count.fetch_add(1, Ordering::SeqCst);
            self.spilled_bytes.fetch_add(total_size, Ordering::SeqCst);
            spills.push(path);
        }
        Ok(total_size)
    }

    fn get_used(&self) -> isize {
        self.used.load(Ordering::SeqCst)
    }

    fn update_used(&mut self, delta: isize) {
        self.used.fetch_add(delta, Ordering::SeqCst);
    }

    fn spilled_bytes(&self) -> usize {
        self.spilled_bytes.load(Ordering::SeqCst)
    }

    fn spilled_count(&self) -> usize {
        self.spilled_count.load(Ordering::SeqCst)
    }
}

async fn merge_sort(
    sorted_bathes: Vec<RecordBatch>,
    schema: SchemaRef,
    expressions: &[PhysicalSortExpr],
    target_batch_size: usize,
    baseline_metrics: BaselineMetrics,
) -> Result<SendableRecordBatchStream> {
    if sorted_bathes.len() == 1 {
        Ok(Box::pin(SizedRecordBatchStream::new(
            schema,
            vec![Arc::new(sorted_bathes(0))],
        )))
    } else {
        let streams = sorted_bathes
            .into_iter()
            .map(|batch| {
                let (mut tx, rx) = futures::channel::mpsc::channel(1);
                tx.send(ArrowResult::Ok(batch)).await;
                rx
            })
            .collect::<Vec<_>>();
        Ok(Box::pin(SortPreservingMergeStream::new(
            streams,
            schema,
            expressions,
            target_batch_size,
            baseline_metrics,
        )))
    }
}

async fn spill(
    mut sorted_stream: Result<SendableRecordBatchStream>,
    path: String,
    schema: SchemaRef,
) -> Result<()> {
    let (mut sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) =
        tokio::sync::mpsc::channel(2);
    while let Some(item) = sorted_stream.next().await {
        sender.send(item).await.ok();
    }
    task::spawn_blocking(move || write_sorted(receiver, path, schema));
    Ok(())
}

fn write_sorted(
    mut receiver: Receiver<RecordBatch>,
    path: String,
    schema: SchemaRef,
) -> Result<()> {
    let mut writer = IPCWriterWrapper::new(path.as_ref(), schema.as_ref())?;
    while let Some(batch) = receiver.blocking_recv() {
        writer.write(&batch)?;
    }
    writer.finish()?;
    info!(
        "Spilled {} batches of total {} rows to disk, memory released {}",
        writer.num_batches, writer.num_rows, writer.num_bytes
    );
    Ok(())
}

struct SpillableSortedStream {
    id: MemoryConsumerId,
    schema: SchemaRef,
    in_mem_batches: Mutex<Vec<RecordBatch>>,
    /// Sort expressions
    expr: Vec<PhysicalSortExpr>,
    runtime: RuntimeEnv,
}

impl SpillableSortedStream {
    fn new() -> Self {
        Self {}
    }

    fn memory_used(&self) -> usize {}

    fn get_sorted_stream(&self) {}

    fn spill_remaining(&self) {}
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
            1 => Ok(Arc::new(ExternalSorter::try_new(
                self.expr.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "SortExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if !self.preserve_partitioning {
            if 0 != partition {
                return Err(DataFusionError::Internal(format!(
                    "SortExec invalid partition {}",
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

        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let input = self.input.execute(partition).await?;

        Ok(Box::pin(SortStream::new(
            input,
            self.expr.clone(),
            baseline_metrics,
        )))
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

pin_project! {
    /// stream for sort plan
    struct ExternalSortStream {
        #[pin]
        output: futures::channel::oneshot::Receiver<ArrowResult<Option<RecordBatch>>>,
        finished: bool,
        schema: SchemaRef,
    }
}

impl ExternalSortStream {
    fn new(
        input: SendableRecordBatchStream,
        expr: Vec<PhysicalSortExpr>,
        baseline_metrics: BaselineMetrics,
    ) -> Self {
        let (tx, rx) = futures::channel::oneshot::channel();
        let schema = input.schema();
        tokio::spawn(async move {
            let schema = input.schema();
            let sorted_batch = common::collect(input)
                .await
                .map_err(DataFusionError::into_arrow_external_error)
                .and_then(move |batches| {
                    let timer = baseline_metrics.elapsed_compute().timer();
                    // combine all record batches into one for each column
                    let combined = common::combine_batches(&batches, schema.clone())?;
                    // sort combined record batch
                    let result = combined
                        .map(|batch| sort_batch(batch, schema, &expr))
                        .transpose()?
                        .record_output(&baseline_metrics);
                    timer.done();
                    Ok(result)
                });

            tx.send(sorted_batch)
        });

        Self {
            output: rx,
            finished: false,
            schema,
        }
    }
}

impl Stream for ExternalSortStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        // is the output ready?
        let this = self.project();
        let output_poll = this.output.poll(cx);

        match output_poll {
            Poll::Ready(result) => {
                *this.finished = true;

                // check for error in receiving channel and unwrap actual result
                let result = match result {
                    Err(e) => {
                        Some(Err(ArrowError::External("".to_string(), Box::new(e))))
                    } // error receiving
                    Ok(result) => result.transpose(),
                };

                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for ExternalSortStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::expressions::col;
    use crate::physical_plan::memory::MemoryExec;
    use crate::physical_plan::{
        collect,
        csv::{CsvExec, CsvReadOptions},
    };
    use crate::test;
    use arrow::array::*;
    use arrow::datatypes::*;

    #[tokio::test]
    async fn test_sort() -> Result<()> {
        let schema = test::aggr_test_schema();
        let partitions = 4;
        let path = test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;
        let csv = CsvExec::try_new(
            &path,
            CsvReadOptions::new().schema(&schema),
            None,
            1024,
            None,
        )?;

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
        let metrics = sort_exec.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
        assert_eq!(metrics.output_rows().unwrap(), 8);
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
