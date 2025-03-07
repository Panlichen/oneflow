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
#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

#include <netdb.h>
#include <arpa/inet.h>

namespace oneflow {

struct IBVerbsCommNetRMADesc {
  uint64_t mem_ptr;
  uint64_t mem_size;
  uint32_t mr_rkey;
};

class IBVerbsCommNet final : public CommNetIf<IBVerbsMemDesc> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsCommNet);
  ~IBVerbsCommNet();

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void RecvActorMsg(const ActorMsg& msg);

 private:
  friend class Singleton<IBVerbsCommNet>;
  IBVerbsCommNet();

  IBVerbsMemDesc* NewMemDesc(void* ptr, size_t byte_size) override {
    return new IBVerbsMemDesc(pd_, ptr, byte_size);
  }

  void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) override;
  void PollCQ();

  static const int32_t max_poll_wc_num_;

  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* cq_;
  std::vector<IBVerbsQP*> qp_vec_;
  std::atomic_flag poll_exit_flag_;
  std::thread poll_thread_;
  HashMap<std::pair<int64_t, uint64_t>, std::shared_ptr<IBVerbsCommNetRMADesc>>
      remote_regst2rma_desc_;
  std::mutex remote_regst2rma_desc_mutex_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_COMM_NETWORK_H_
