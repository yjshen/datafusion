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

//! DynMutableListArray from arrow/io/avro/read/nested.rs

use arrow::array::{Array, ListArray, MutableArray, Offset};
use arrow::bitmap::MutableBitmap;
use arrow::buffer::MutableBuffer;
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use std::sync::Arc;

/// Auxiliary struct
#[derive(Debug)]
pub struct DynMutableListArray<O: Offset> {
    data_type: DataType,
    offsets: MutableBuffer<O>,
    values: Box<dyn MutableArray>,
    validity: Option<MutableBitmap>,
}

impl<O: Offset> DynMutableListArray<O> {
    pub fn new_from(
        values: Box<dyn MutableArray>,
        data_type: DataType,
        capacity: usize,
    ) -> Self {
        let mut offsets = MutableBuffer::<O>::with_capacity(capacity + 1);
        offsets.push(O::default());
        assert_eq!(values.len(), 0);
        ListArray::<O>::get_child_field(&data_type);
        Self {
            data_type,
            offsets,
            values,
            validity: None,
        }
    }

    /// Creates a new [`MutableListArray`] from a [`MutableArray`] and capacity.
    pub fn new_with_capacity(values: Box<dyn MutableArray>, capacity: usize) -> Self {
        let data_type = ListArray::<O>::default_datatype(values.data_type().clone());
        Self::new_from(values, data_type, capacity)
    }

    /// The values
    #[allow(dead_code)]
    pub fn mut_values(&mut self) -> &mut dyn MutableArray {
        self.values.as_mut()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn try_push_valid(&mut self) -> Result<(), ArrowError> {
        let size = self.values.len();
        let size = O::from_usize(size).ok_or(ArrowError::KeyOverflowError)?; // todo: make this error
        assert!(size >= *self.offsets.last().unwrap());

        self.offsets.push(size);
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    fn push_null(&mut self) {
        self.offsets.push(self.last_offset());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    #[inline]
    fn last_offset(&self) -> O {
        *self.offsets.last().unwrap()
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len() - 1;

        let mut validity = MutableBitmap::new();
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }
}

impl<O: Offset> MutableArray for DynMutableListArray<O> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(ListArray::from_data(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_arc(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        ))
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(ListArray::from_data(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_arc(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        ))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn shrink_to_fit(&mut self) {
        todo!();
    }
}
