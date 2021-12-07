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

//! Shuffle related

pub mod shuffle_repartition;

use crate::error::{DataFusionError, Result};
use arrow::error::Result as ArrowResult;
use arrow::io::ipc::read::{read_file_segment_metadata, FileMetadata, FileReader};
use arrow::record_batch::RecordBatch;
use std::io::{Read, Seek, SeekFrom};

fn read_meta<R: Read + Seek>(
    reader: &mut R,
    end: usize,
) -> Result<(usize, FileMetadata)> {
    let ipc_end = end - 4;
    reader.seek(SeekFrom::Start(end as u64))?;
    let mut meta_buf = [0; 4];
    reader.read_exact(&mut meta_buf)?;
    let ipc_length = i32::from_le_bytes(meta_buf);
    let ipc_start = end - ipc_length as usize;

    let ipc_meta = read_file_segment_metadata(reader, ipc_start as u64, ipc_end as u64)
        .map_err(DataFusionError::ArrowError)?;
    Ok((ipc_start, ipc_meta))
}

pub struct FileSegment<R: Read + Seek> {
    start: usize,
    current_start: usize,
    reader: Option<FileReader<R>>,
}

impl<R: Read + Seek> FileSegment<R> {
    fn new(mut reader: R, start: usize, end: usize) -> Result<Self> {
        let (current_start, meta) = read_meta(&mut reader, end)?;
        Ok(Self {
            start,
            current_start,
            reader: Some(FileReader::new(reader, meta, None)),
        })
    }
}

impl<R: Read + Seek> Iterator for FileSegment<R> {
    type Item = ArrowResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.reader.as_mut().unwrap().next();
        match next {
            None => {
                if self.current_start == self.start {
                    None
                } else {
                    let mut r = self.reader.take().unwrap().into_inner();
                    let (new_start, meta) =
                        read_meta(&mut r, self.current_start).unwrap();
                    self.reader = Some(FileReader::new(r, meta, None));
                    self.current_start = new_start;
                    self.next()
                }
            }
            item => item,
        }
    }
}
