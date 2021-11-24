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

//! Defines the execution plan for the sort aggregate operation

use std::any::Any;
use std::sync::Arc;

use futures::stream::StreamExt;

use crate::execution::runtime_env::RUNTIME_ENV;
use crate::physical_plan::{
    Accumulator, AggregateExpr, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, PhysicalExpr,
};
use crate::{
    error::{DataFusionError, Result},
    scalar::ScalarValue,
};

use arrow::{
    array::*,
    datatypes::{Schema, SchemaRef},
    error::Result as ArrowResult,
    record_batch::RecordBatch,
};

use async_trait::async_trait;

use crate::execution::runtime_env::RuntimeEnv;
use crate::physical_plan::aggregations::{
    aggregate_expressions, create_accumulators, evaluate_many, AggregateMode,
};
use crate::physical_plan::expressions::{exprs_to_sort_columns, PhysicalSortExpr};
use crate::physical_plan::metrics::{
    self, BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::sorts::external_sort::ExternalSortExec;
use crate::physical_plan::stream::RecordBatchReceiverStream;
use crate::physical_plan::Statistics;
use crate::physical_plan::{expressions::Column, SendableRecordBatchStream};
use arrow::compute::partition::lexicographical_partition_ranges;
use arrow::datatypes::Field;
use std::ops::Range;
use tokio::sync::mpsc::{Receiver, Sender};

/// Sort aggregate execution plan
#[derive(Debug)]
pub struct SortAggregateExec {
    /// Aggregation mode (full, partial)
    mode: AggregateMode,
    /// Grouping expressions
    group_expr: Vec<Column>,
    /// Aggregate expressions
    aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    /// Input plan, could be a partial aggregate or the input to the aggregate
    input: Arc<dyn ExecutionPlan>,
    /// Schema after the aggregate is applied
    schema: SchemaRef,
    /// Input schema before any aggregation is applied. For partial aggregate this will be the
    /// same as input.schema() but for the final aggregate it will be the same as the input
    /// to the partial aggregate
    input_schema: SchemaRef,
    /// Execution Metrics
    metrics: ExecutionPlanMetricsSet,
}

fn create_schema(
    input_schema: &Schema,
    group_expr: &[Column],
    aggr_expr: &[Arc<dyn AggregateExpr>],
    mode: AggregateMode,
) -> Result<Schema> {
    let mut fields = Vec::with_capacity(group_expr.len() + aggr_expr.len());
    for c in group_expr {
        fields.push(Field::new(
            c.name(),
            c.data_type(input_schema)?,
            c.nullable(input_schema)?,
        ))
    }

    match mode {
        AggregateMode::Partial => {
            // in partial mode, the fields of the accumulator's state
            for expr in aggr_expr {
                fields.extend(expr.state_fields()?.iter().cloned())
            }
        }
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            // in final mode, the field with the final result of the accumulator
            for expr in aggr_expr {
                fields.push(expr.field()?)
            }
        }
    }

    Ok(Schema::new(fields))
}

impl SortAggregateExec {
    /// Create a new Sort aggregate execution plan
    pub fn try_new(
        mode: AggregateMode,
        group_expr: Vec<Column>,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
    ) -> Result<Self> {
        let schema = create_schema(&input.schema(), &group_expr, &aggr_expr, mode)?;

        let schema = Arc::new(schema);

        Ok(SortAggregateExec {
            mode,
            group_expr,
            aggr_expr,
            input,
            schema,
            input_schema,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Aggregation mode (full, partial)
    pub fn mode(&self) -> &AggregateMode {
        &self.mode
    }

    /// Grouping expressions
    pub fn group_expr(&self) -> &[Column] {
        &self.group_expr
    }

    /// Aggregate expressions
    pub fn aggr_expr(&self) -> &[Arc<dyn AggregateExpr>] {
        &self.aggr_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any aggregates are applied
    pub fn input_schema(&self) -> SchemaRef {
        self.input_schema.clone()
    }
}

#[async_trait]
impl ExecutionPlan for SortAggregateExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn required_child_distribution(&self) -> Distribution {
        match &self.mode {
            AggregateMode::Partial => Distribution::UnspecifiedDistribution,
            AggregateMode::FinalPartitioned => Distribution::HashPartitioned(
                self.group_expr
                    .iter()
                    .map(|x| Arc::new(x.clone()) as Arc<dyn PhysicalExpr>)
                    .collect::<Vec<_>>(),
            ),
            AggregateMode::Final => Distribution::SinglePartition,
        }
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(SortAggregateExec::try_new(
                self.mode,
                self.group_expr.clone(),
                self.aggr_expr.clone(),
                children[0].clone(),
                self.input_schema.clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "HashAggregateExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition).await?;
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let sort = self
            .input
            .as_any()
            .downcast_ref::<ExternalSortExec>()
            .unwrap()
            .expr()
            .iter()
            .map(|s| s.clone())
            .collect::<Vec<_>>();

        assert!(
            !self.group_expr.is_empty(),
            "Should use hash_aggregate for non_grouping case"
        );

        let (tx, rx): (
            Sender<ArrowResult<RecordBatch>>,
            Receiver<ArrowResult<RecordBatch>>,
        ) = tokio::sync::mpsc::channel(2);

        let mut driver = SortAggregateDriver::new(
            input,
            sort,
            self.schema.clone(),
            RUNTIME_ENV.clone(),
            self.group_expr.clone(),
            self.aggr_expr.clone(),
            self.mode,
        )?;

        tokio::spawn(async move {
            driver.aggregate(&tx).await?;
            Ok::<(), DataFusionError>(())
        });

        let result = RecordBatchReceiverStream::create(&self.schema, rx);
        Ok(result)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "SortAggregateExec: mode={:?}", self.mode)?;
                let g: Vec<String> =
                    self.group_expr.iter().map(|e| e.to_string()).collect();
                write!(f, ", gby=[{}]", g.join(", "))?;

                let a: Vec<String> = self
                    .aggr_expr
                    .iter()
                    .map(|agg| agg.name().to_string())
                    .collect();
                write!(f, ", aggr=[{}]", a.join(", "))?;
            }
        }
        Ok(())
    }

    fn statistics(&self) -> Statistics {
        // TODO stats: group expressions:
        // - once expressions will be able to compute their own stats, use it here
        // - case where we group by on a column for which with have the `distinct` stat
        match self.mode {
            _ => Statistics::default(),
        }
    }
}

/// The state that is built for the current output group.
#[derive(Debug)]
struct GroupState {
    /// The actual group by values, one for each group column
    group_by_values: Vec<ScalarValue>,

    // Accumulator state, one for each aggregate
    accumulator_set: Vec<Box<dyn Accumulator>>,
}

struct OutputBuffer {
    /// State for each group
    group_states: Vec<GroupState>,
    batch_size: usize,
}

impl OutputBuffer {
    fn new(batch_size: usize) -> Self {
        Self {
            group_states: vec![],
            batch_size,
        }
    }

    fn is_full(&self) -> bool {
        self.group_states.len() >= self.batch_size
    }

    fn is_empty(&self) -> bool {
        self.group_states.len() == 0
    }

    fn output(
        &mut self,
        mode: &AggregateMode,
        output_schema: &Arc<Schema>,
    ) -> ArrowResult<RecordBatch> {
        let accs = &self.group_states[0].accumulator_set;
        let num_group_expr = self.group_states[0].group_by_values.len();
        let mut acc_data_types: Vec<usize> = vec![];

        // Calculate number/shape of state arrays
        match mode {
            AggregateMode::Partial => {
                for acc in accs.iter() {
                    let state = acc
                        .state()
                        .map_err(DataFusionError::into_arrow_external_error)?;
                    acc_data_types.push(state.len());
                }
            }
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                acc_data_types = vec![1; accs.len()];
            }
        }

        let mut columns = (0..num_group_expr)
            .map(|i| {
                ScalarValue::iter_to_array(
                    self.group_states
                        .iter()
                        .map(|group_state| group_state.group_by_values[i].clone()),
                )
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|x| x.into_arrow_external_error())?;

        // add state / evaluated arrays
        for (x, &state_len) in acc_data_types.iter().enumerate() {
            for y in 0..state_len {
                match mode {
                    AggregateMode::Partial => {
                        let res = ScalarValue::iter_to_array(
                            self.group_states.iter().map(|group_state| {
                                let x = group_state.accumulator_set[x].state().unwrap();
                                x[y].clone()
                            }),
                        )
                        .map_err(DataFusionError::into_arrow_external_error)?;
                        columns.push(res);
                    }
                    AggregateMode::Final | AggregateMode::FinalPartitioned => {
                        let res = ScalarValue::iter_to_array(
                            self.group_states.iter().map(|group_state| {
                                group_state.accumulator_set[x].evaluate().unwrap()
                            }),
                        )
                        .map_err(DataFusionError::into_arrow_external_error)?;
                        columns.push(res);
                    }
                }
            }
        }

        // cast output if needed (e.g. for types like Dictionary where
        // the intermediate GroupByScalar type was not the same as the
        // output
        let columns = columns
            .iter()
            .zip(output_schema.fields().iter())
            .map(|(col, desired_field)| {
                arrow::compute::cast::cast(col.as_ref(), desired_field.data_type())
                    .map(Arc::from)
            })
            .collect::<ArrowResult<Vec<_>>>()?;

        let result = RecordBatch::try_new(output_schema.clone(), columns);
        self.group_states.drain(..);
        result
    }
}

struct SortAggregateDriver {
    input: SendableRecordBatchStream,
    output: OutputBuffer,
    state_idx: usize,
    needs_new_state: bool,
    sort: Vec<PhysicalSortExpr>,
    group_expr: Vec<Column>,
    aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    schema: Arc<Schema>,
    mode: AggregateMode,
}

impl SortAggregateDriver {
    fn new(
        input: SendableRecordBatchStream,
        sort: Vec<PhysicalSortExpr>,
        schema: Arc<Schema>,
        runtime: Arc<RuntimeEnv>,
        group_expr: Vec<Column>,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        mode: AggregateMode,
    ) -> Result<Self> {
        let batch_size = runtime.batch_size();
        let aggregate_expressions =
            aggregate_expressions(&aggr_expr, &mode, group_expr.len())?;
        Ok(Self {
            input,
            output: OutputBuffer::new(batch_size),
            state_idx: 0,
            needs_new_state: true,
            sort,
            group_expr,
            aggr_expr,
            aggregate_expressions,
            schema,
            mode,
        })
    }

    async fn aggregate(
        &mut self,
        sender: &Sender<ArrowResult<RecordBatch>>,
    ) -> Result<()> {
        while let Some(batch) = self.input.next().await {
            let batch = batch.map_err(DataFusionError::ArrowError)?;
            let num_rows_in_batch = batch.num_rows();

            let columns = exprs_to_sort_columns(&batch, &self.sort)?;
            let columns = &columns.iter().map(|x| x.into()).collect::<Vec<_>>();
            let ranges = lexicographical_partition_ranges(columns)?.collect::<Vec<_>>();

            let groups = self
                .group_expr
                .iter()
                .map(|c| batch.column(c.index()).clone())
                .collect::<Vec<_>>();
            let agg_inputs: Vec<Vec<ArrayRef>> =
                evaluate_many(&self.aggregate_expressions, &batch)?;

            let mut is_first = true;
            for range in ranges {
                let is_last = range.end == num_rows_in_batch;
                if is_first && is_last {
                    if self.cmp_key_with_state(&groups) {
                        self.update_state(&agg_inputs, &range)?;
                    } else {
                        self.needs_new_state = true;
                        self.create_new_state(&groups, range.start)?;
                        self.update_state(&agg_inputs, &range)?;
                    }
                } else if is_first {
                    if self.cmp_key_with_state(&groups) {
                        self.update_state(&agg_inputs, &range)?;
                        self.needs_new_state = true;
                    } else {
                        self.needs_new_state = true;
                        self.create_new_state(&groups, range.start)?;
                        self.update_state(&agg_inputs, &range)?;
                        self.needs_new_state = true;
                    }
                    is_first = false;
                } else if is_last {
                    self.create_new_state(&groups, range.start)?;
                    self.update_state(&agg_inputs, &range)?;
                } else {
                    // create new state and output
                    self.create_new_state(&groups, range.start)?;
                    self.update_state(&agg_inputs, &range)?;
                    self.needs_new_state = true;
                }
            }

            if self.output.is_full() {
                let result = self.output.output(&self.mode, &self.schema);
                if let Err(e) = sender.send(result).await {
                    println!("ERROR batch via aggregation stream: {}", e);
                };
            }
        }

        // send output batch
        if !self.output.is_empty() {
            let result = self.output.output(&self.mode, &self.schema);
            if let Err(e) = sender.send(result).await {
                println!("ERROR batch via aggregation stream last batch: {}", e);
            };
        }

        Ok(())
    }

    /// true for equals
    fn cmp_key_with_state(&self, grps: &[ArrayRef]) -> bool {
        if self.needs_new_state {
            return false;
        }

        grps.iter()
            .zip(
                self.output.group_states[self.state_idx]
                    .group_by_values
                    .iter(),
            )
            // we are always comparing first row in batch with state,
            // since it's the only case that needs extra comparison.
            .all(|(array, scalar)| scalar.eq_array(array, 0))
    }

    fn update_state(
        &mut self,
        agg_inputs: &Vec<Vec<ArrayRef>>,
        range: &Range<usize>,
    ) -> Result<()> {
        let agg_mod = self.mode;
        self.output.group_states[self.state_idx]
            .accumulator_set
            .iter_mut()
            .zip(agg_inputs.iter())
            .try_for_each(|(accumulator, values)| {
                let sliced = values
                    .iter()
                    .map(|v| ArrayRef::from(v.slice(range.start, range.len())))
                    .collect::<Vec<ArrayRef>>();
                match agg_mod {
                    AggregateMode::Partial => accumulator.update_batch(&sliced),
                    AggregateMode::FinalPartitioned | AggregateMode::Final => {
                        // note: the aggregation here is over states, not values, thus the merge
                        accumulator.merge_batch(&values)
                    }
                }
            })?;
        Ok(())
    }

    fn create_new_state(&mut self, grps: &[ArrayRef], row: usize) -> Result<()> {
        assert!(self.needs_new_state);
        let group_by_values = grps
            .iter()
            .map(|col| ScalarValue::try_from_array(col, row))
            .collect::<Result<Vec<_>>>()?;
        let accumulator_set = create_accumulators(&self.aggr_expr)?;
        let new_state = GroupState {
            group_by_values,
            accumulator_set,
        };
        self.output.group_states.push(new_state);
        self.state_idx += 1;
        self.needs_new_state = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use arrow::array::{Float64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field};

    use super::*;
    use crate::physical_plan::expressions::{col, Avg};
    use crate::{assert_batches_sorted_eq, physical_plan::common};

    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::RecordBatchStream;
    use futures::task::{Context, Poll};
    use futures::Stream;

    /// some mock data to aggregates
    fn some_data() -> (Arc<Schema>, Vec<RecordBatch>) {
        // define a schema.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::Float64, false),
        ]));

        // define data.
        (
            schema.clone(),
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(UInt32Array::from_slice(&[2, 3, 4, 4])),
                        Arc::new(Float64Array::from_slice(&[1.0, 2.0, 3.0, 4.0])),
                    ],
                )
                .unwrap(),
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(UInt32Array::from_slice(&[2, 3, 3, 4])),
                        Arc::new(Float64Array::from_slice(&[1.0, 2.0, 3.0, 4.0])),
                    ],
                )
                .unwrap(),
            ],
        )
    }

    /// build the aggregates on the data from some_data() and check the results
    async fn check_aggregates(input: Arc<dyn ExecutionPlan>) -> Result<()> {
        let input_schema = input.schema();

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("a", &input_schema)?, "a".to_string())];

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Avg::new(
            col("b", &input_schema)?,
            "AVG(b)".to_string(),
            DataType::Float64,
        ))];

        let partial_aggregate = Arc::new(SortAggregateExec::try_new(
            AggregateMode::Partial,
            groups.clone(),
            aggregates.clone(),
            input,
            input_schema.clone(),
        )?);

        let result = common::collect(partial_aggregate.execute(0).await?).await?;

        let expected = vec![
            "+---+---------------+-------------+",
            "| a | AVG(b)[count] | AVG(b)[sum] |",
            "+---+---------------+-------------+",
            "| 2 | 2             | 2           |",
            "| 3 | 3             | 7           |",
            "| 4 | 3             | 11          |",
            "+---+---------------+-------------+",
        ];
        assert_batches_sorted_eq!(expected, &result);

        let merge = Arc::new(CoalescePartitionsExec::new(partial_aggregate));

        let final_group: Vec<Arc<dyn PhysicalExpr>> = (0..groups.len())
            .map(|i| col(&groups[i].1, &input_schema))
            .collect::<Result<_>>()?;

        let merged_aggregate = Arc::new(SortAggregateExec::try_new(
            AggregateMode::Final,
            final_group
                .iter()
                .enumerate()
                .map(|(i, expr)| (expr.clone(), groups[i].1.clone()))
                .collect(),
            aggregates,
            merge,
            input_schema,
        )?);

        let result = common::collect(merged_aggregate.execute(0).await?).await?;
        assert_eq!(result.len(), 1);

        let batch = &result[0];
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let expected = vec![
            "+---+--------------------+",
            "| a | AVG(b)             |",
            "+---+--------------------+",
            "| 2 | 1                  |",
            "| 3 | 2.3333333333333335 |", // 3, (2 + 3 + 2) / 3
            "| 4 | 3.6666666666666665 |", // 4, (3 + 4 + 4) / 3
            "+---+--------------------+",
        ];

        assert_batches_sorted_eq!(&expected, &result);

        let metrics = merged_aggregate.metrics().unwrap();
        let output_rows = metrics.output_rows().unwrap();
        assert_eq!(3, output_rows);

        Ok(())
    }

    /// Define a test source that can yield back to runtime before returning its first item ///

    #[derive(Debug)]
    struct TestYieldingExec {
        /// True if this exec should yield back to runtime the first time it is polled
        pub yield_first: bool,
    }

    #[async_trait]
    impl ExecutionPlan for TestYieldingExec {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn schema(&self) -> SchemaRef {
            some_data().0
        }

        fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn output_partitioning(&self) -> Partitioning {
            Partitioning::UnknownPartitioning(1)
        }

        fn with_new_children(
            &self,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::Internal(format!(
                "Children cannot be replaced in {:?}",
                self
            )))
        }

        async fn execute(&self, _partition: usize) -> Result<SendableRecordBatchStream> {
            let stream;
            if self.yield_first {
                stream = TestYieldingStream::New;
            } else {
                stream = TestYieldingStream::Yielded;
            }
            Ok(Box::pin(stream))
        }

        fn statistics(&self) -> Statistics {
            let (_, batches) = some_data();
            common::compute_record_batch_statistics(&[batches], &self.schema(), None)
        }
    }

    /// A stream using the demo data. If inited as new, it will first yield to runtime before returning records
    enum TestYieldingStream {
        New,
        Yielded,
        ReturnedBatch1,
        ReturnedBatch2,
    }

    impl Stream for TestYieldingStream {
        type Item = ArrowResult<RecordBatch>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match &*self {
                TestYieldingStream::New => {
                    *(self.as_mut()) = TestYieldingStream::Yielded;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                TestYieldingStream::Yielded => {
                    *(self.as_mut()) = TestYieldingStream::ReturnedBatch1;
                    Poll::Ready(Some(Ok(some_data().1[0].clone())))
                }
                TestYieldingStream::ReturnedBatch1 => {
                    *(self.as_mut()) = TestYieldingStream::ReturnedBatch2;
                    Poll::Ready(Some(Ok(some_data().1[1].clone())))
                }
                TestYieldingStream::ReturnedBatch2 => Poll::Ready(None),
            }
        }
    }

    impl RecordBatchStream for TestYieldingStream {
        fn schema(&self) -> SchemaRef {
            some_data().0
        }
    }

    //// Tests ////

    #[tokio::test]
    async fn aggregate_source_not_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: false });

        check_aggregates(input).await
    }

    #[tokio::test]
    async fn aggregate_source_with_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: true });

        check_aggregates(input).await
    }
}
