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
#include "oneflow/core/job/of_collective_boxing/of_request_store.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace boxing {

namespace of_collective {

OfRequestEntry::OfRequestEntry(const RequestDesc& desc, int coll_id) : desc_(desc), coll_id_(coll_id) {
  std::set<int64_t> node_ids;
  for (int64_t global_rank = 0; global_rank < desc.device_set().device().size(); ++global_rank) {
    const DeviceDesc& device = desc.device_set().device(global_rank);
    if (device.machine_id() == GlobalProcessCtx::Rank()) {
      local_device_vec_.emplace_back(device);
      global_rank2local_rank_.emplace(global_rank, local_rank2global_rank_.size());
      local_rank2global_rank_.emplace_back(global_rank);
    }
    node_ids.emplace(device.machine_id());
  }
  node_count_ = node_ids.size();
  elem_cnt_ = Shape(desc.op_desc().shape()).elem_cnt();
  size_in_bytes_ = elem_cnt_ * GetSizeOfDataType(desc.op_desc().data_type());
  device_set_symbol_.reset(desc.device_set());

  FOR_EACH(id7nego_tree_info, desc.negotiation_tree_topo()) {
    auto nego_tree_info = RuntimeNegoTreeInfo();
    nego_tree_info.upstream_id = id7nego_tree_info->second.upstream_id();
    nego_tree_info.downstream_id = PbRf2StdVec(id7nego_tree_info->second.downstream_id());
    nego_tree_topo_[id7nego_tree_info->first] = std::move(nego_tree_info);
  }
}

void OfRequestStore::InitJob(int64_t job_id, const RequestSet& request_set) {
  std::vector<std::unique_ptr<OfRequestEntry>>& request_entry_vec = job_id2request_entry_vec_[job_id];
  CHECK_EQ(request_entry_vec.size(), 0);
  for (const RequestDesc& desc : request_set.request()) {
    request_entry_vec.emplace_back(std::make_unique<OfRequestEntry>(desc, coll_id_counter_++));
  }
  
  job_id2index_to_issue[job_id] = 0;
  job_id2curr_coll_id_vec[job_id] = 0;
  for (int vec_num = 0; vec_num < NUM_COLL_ID_VEC; ++vec_num) {
    job_id2ordered_local_coll_ids[job_id].emplace_back(std::vector<int>());
    job_id2local_coll_id2index[job_id].emplace_back(HashMap<int, int>());
  }

  for (int32_t i = 0; i < request_entry_vec.size(); ++i) {
    const std::unique_ptr<OfRequestEntry>& entry = request_entry_vec.at(i);
    if (entry->HasRankOnThisNode()) {
      int entry_coll_id = entry->coll_id();

      for (int vec_num = 0; vec_num < NUM_COLL_ID_VEC; ++vec_num) {
        std::vector<int>& ordered_local_coll_ids = job_id2ordered_local_coll_ids[job_id][vec_num];
        HashMap<int, int>& local_coll_id2index = job_id2local_coll_id2index[job_id][vec_num];

        local_coll_id2index.emplace(entry_coll_id, ordered_local_coll_ids.size());
        ordered_local_coll_ids.emplace_back(entry_coll_id);
      }
    }

    // TODO(Panlichen): OfRequestId大概可以删了。
    CHECK(name2request_id_.emplace(entry->desc().op_desc().name(), OfRequestId(job_id, i)).second);
  }
  // for (int32_t i = 0; i < ordered_local_coll_ids.size(); ++i) {
  //   VLOG(1) << "job_id2ordered_local_coll_ids[" << job_id << "][" << i << "] = " << job_id2ordered_local_coll_ids[job_id][i];
  //   VLOG(1) << "job_id2local_coll_id2index[" << job_id << "][" << job_id2ordered_local_coll_ids[job_id][i] << "] = " << job_id2local_coll_id2index[job_id][job_id2ordered_local_coll_ids[job_id][i]];
  // }
}

void OfRequestStore::DeinitJob(int64_t job_id) {
  const auto& it = job_id2request_entry_vec_.find(job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  const auto& request_entry_vec = it->second;
  for (const auto& request_entry : request_entry_vec) {
    name2request_id_.erase(request_entry->desc().op_desc().name());
  }
  job_id2request_entry_vec_.erase(job_id);
}

struct OfRequestEntryToken {
  OfRequestEntry* request_entry;
};

void* OfRequestStore::CreateOfRequestEntryToken(const OfRequestId& request_id) {
  auto it = job_id2request_entry_vec_.find(request_id.job_id);
  CHECK(it != job_id2request_entry_vec_.end());
  return new OfRequestEntryToken{it->second.at(request_id.request_index).get()};
}

void OfRequestStore::DestroyOfRequestEntryToken(void* request_entry_token) {
  auto token = static_cast<OfRequestEntryToken*>(request_entry_token);
  delete token;
}

OfRequestEntry* OfRequestStore::GetOfRequestEntry(void* request_entry_token) {
  return static_cast<OfRequestEntryToken*>(request_entry_token)->request_entry;
}

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow
