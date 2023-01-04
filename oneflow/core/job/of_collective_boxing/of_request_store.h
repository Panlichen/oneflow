/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applinamespace collectivecable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_OF_REQUEST_STORE_H_
#define ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_OF_REQUEST_STORE_H_

#include <memory>
#include <queue>
#include <mutex>
#include <nccl.h>

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"

namespace oneflow {

namespace boxing {

namespace of_collective {

template<class T, class Compare = std::greater<T>, class Container = std::vector<T> > // 默认小顶堆
using Heap = std::priority_queue<T, Container, Compare>;

struct RuntimeNegoTreeInfo {
  int64_t upstream_id;
  std::vector<int64_t> downstream_id;
};

class OfRequestEntry final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfRequestEntry);
  OfRequestEntry(const RequestDesc& desc, int coll_id);
  ~OfRequestEntry() = default;

  const RequestDesc& desc() const { return desc_; }
  int32_t LocalRankCount() const { return local_rank2global_rank_.size(); }
  int32_t LocalRankToGlobalRank(int32_t local_rank) const {
    return local_rank2global_rank_.at(local_rank);
  }
  int32_t GlobalRankToLocalRank(int32_t global_rank) const {
    return global_rank2local_rank_.at(global_rank);
  }
  bool HasRankOnThisNode() const { return !local_rank2global_rank_.empty(); }
  int32_t NodeCount() const { return node_count_; }
  const DeviceDesc& LocalDeviceDesc(int32_t local_rank) const {
    return local_device_vec_.at(local_rank);
  }
  bool IsRootOnThisNode() const {
    return (!local_rank2global_rank_.empty()) && local_rank2global_rank_.front() == 0;
  }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t size_in_bytes() const { return size_in_bytes_; }
  const Symbol<DeviceSet>& device_set_symbol() const { return device_set_symbol_; }

  std::map<int64_t, RuntimeNegoTreeInfo> nego_tree_topo() const { return nego_tree_topo_; }

  int coll_id() const { return coll_id_; }

 private:
  RequestDesc desc_;
  int32_t node_count_;
  std::vector<DeviceDesc> local_device_vec_;
  std::vector<int64_t> local_rank2global_rank_;
  std::map<int64_t, int64_t> global_rank2local_rank_;
  int64_t elem_cnt_;
  int64_t size_in_bytes_;

  std::map<int64_t, RuntimeNegoTreeInfo> nego_tree_topo_;

  Symbol<DeviceSet> device_set_symbol_;

  // 一个request绑定一个coll_id。
  int coll_id_;
};

struct OfRequestId {
  OfRequestId(int64_t job_id, int32_t request_index) : job_id(job_id), request_index(request_index) {}
  int64_t job_id;
  int32_t request_index;
};

struct OfIssueParams {
  int coll_id;
  const void* send_buff;
  void* recv_buff;
  const RankDesc& rank_desc;
  void *cb_args;
  CallbackFunc cb_func;
  ofcclRankCtx_t ofccl_rank_ctx;

  OfIssueParams(int coll_id, const void* send_buff, void* recv_buff, const RankDesc& rank_desc, void *cb_args, CallbackFunc cb_func, ofcclRankCtx_t ofccl_rank_ctx) : coll_id(coll_id), send_buff(send_buff), recv_buff(recv_buff), rank_desc(rank_desc), cb_args(cb_args), cb_func(cb_func), ofccl_rank_ctx(ofccl_rank_ctx) {}
};

class GreaterOfIssueParams {
 public:
  bool operator() (const std::shared_ptr<OfIssueParams>& p1, const std::shared_ptr<OfIssueParams>& p2) {
    return p1->coll_id > p2->coll_id;
  }
};
class LessOfIssueParams {
 public:
  bool operator() (const std::shared_ptr<OfIssueParams>& p1, const std::shared_ptr<OfIssueParams>& p2) {
    return p1->coll_id < p2->coll_id;
  }
};

class OfRequestStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfRequestStore);
  OfRequestStore() : coll_id_counter_(0) {}
  ~OfRequestStore() = default;

  void InitJob(int64_t job_id, const RequestSet& request_set);
  void DeinitJob(int64_t job_id);

  OfRequestEntry* MutOfRequestEntry(const OfRequestId& request_id) {
    auto it = job_id2request_entry_vec_.find(request_id.job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    return it->second.at(request_id.request_index).get();
  }

  void ForEachMutOfRequestEntryForIdsInJob(
      const std::vector<OfRequestId>& request_ids,
      const std::function<void(OfRequestEntry*, int32_t i, const OfRequestId& request_id)>& Handler) {
    if (request_ids.size() == 0) { return; }
    int64_t job_id = request_ids.front().job_id;
    auto it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    for (int32_t i = 0; i < request_ids.size(); ++i) {
      CHECK_EQ(request_ids.at(i).job_id, job_id);
      Handler(it->second.at(request_ids.at(i).request_index).get(), i, request_ids.at(i));
    }
  }

  void ForEachMutOfRequestEntryInJob(
      int64_t job_id,
      const std::function<void(OfRequestEntry*, int32_t i, const OfRequestId& request_id)>& Handler) {
    auto it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    for (int32_t i = 0; i < it->second.size(); ++i) {
      OfRequestId request_id(job_id, i);
      Handler(it->second.at(i).get(), i, request_id);
    }
  }

  int32_t RequestCountForJob(int64_t job_id) const {
    const auto& it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    return it->second.size();
  }

  OfRequestId GetOfRequestIdByName(const std::string& name) const {
    return name2request_id_.at(name);
  }

  void* CreateOfRequestEntryToken(const OfRequestId& request_id);

  void DestroyOfRequestEntryToken(void* token);

  OfRequestEntry* GetOfRequestEntry(void* token);

  // TODO(Panlichen): 考虑支持多个job时的相应调整。
  HashMap<int64_t, std::vector<int>> job_id2ordered_local_coll_ids;
  // HashMap<int64_t, Heap<std::shared_ptr<OfIssueParams>, GreaterOfIssueParams>> job_id2params_heap_in_one_iter; // 小顶堆
  HashMap<int64_t, Heap<std::shared_ptr<OfIssueParams>, LessOfIssueParams>> job_id2params_heap_in_one_iter; // 大顶堆
  HashMap<int64_t, int> job_id2index_to_issue;
  HashMap<int64_t, std::mutex> job_id2heap_mutex;

 private:
  HashMap<int64_t, std::vector<std::unique_ptr<OfRequestEntry>>> job_id2request_entry_vec_;
  HashMap<std::string, OfRequestId> name2request_id_;

  // 在RequestStore里维护跨越job边界的coll_id
  // TODO(Panlichen): 支持多job的话，也考虑使用request_desc->order()作为job内部的coll_id。
  int coll_id_counter_;
};

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_OF_REQUEST_STORE_H_
