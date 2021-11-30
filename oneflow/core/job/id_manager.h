/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_ID_MANAGER_H_
#define ONEFLOW_CORE_JOB_ID_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/graph/task_id_generator.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  int64_t NewRegstDescId() { return regst_desc_id_count_++; }
  int64_t NewMemBlockId() { return mem_block_id_count_++; }
  int64_t NewChunkId() { return chunk_id_count_++; }

  TaskIdGenerator* GetTaskIdGenerator() { return &task_id_gen_; }

 private:
  friend class Global<IDMgr>;
  IDMgr();

  int64_t gpu_device_num_;
  int64_t cpu_device_num_;
  int64_t regst_desc_id_count_;
  int64_t mem_block_id_count_;
  int64_t chunk_id_count_;
  TaskIdGenerator task_id_gen_;

  //  64 bit id design:
  //   sign | machine | thread | local_work_stream | task
  //    1   |   10    |   11   |       21          |  21
  static const int64_t machine_id_bit_num_ = 10;
  static const int64_t thread_id_bit_num_ = 11;
  static const int64_t local_work_stream_id_bit_num_ = 21;
  static const int64_t task_id_bit_num_ = 21;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
