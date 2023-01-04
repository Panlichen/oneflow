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
#include <glog/logging.h>
#include <nccl.h>
#include <cstdint>
#include <memory>
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/of_collective_boxing/collective_manager.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/lazy/actor/of_collective_boxing_actor_context.h"
#include "oneflow/core/lazy/actor/actor_message.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"

namespace oneflow {

using namespace boxing::of_collective;

namespace {

OfCollectiveBoxingActorContext* GetOfCollectiveBoxingActorContext(KernelContext* kernel_ctx) {
  auto* actor_context_provider = CHECK_NOTNULL(dynamic_cast<ActorContextProvider*>(kernel_ctx));
  return CHECK_NOTNULL(
      dynamic_cast<OfCollectiveBoxingActorContext*>(actor_context_provider->GetActorContext()));
}

class OfCollectiveBoxingKernelState final : public KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingKernelState);
  explicit OfCollectiveBoxingKernelState(const RankDesc& rank_desc)
      : coll_id_(Singleton<CollectiveMgr>::Get()->KernelGetCollId(rank_desc)),
        ofccl_rank_ctx_(Singleton<CollectiveMgr>::Get()->KernelGetOfcclRankCtx(rank_desc.rank())) {}
  ~OfCollectiveBoxingKernelState() = default;

  int coll_id() { return coll_id_; }
  ofcclRankCtx_t ofccl_rank_ctx() { return ofccl_rank_ctx_; }

 private:
  int coll_id_;
  ofcclRankCtx_t ofccl_rank_ctx_;
};

class OfCollectiveBoxingGenericKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingGenericKernel);
  OfCollectiveBoxingGenericKernel() = default;
  ~OfCollectiveBoxingGenericKernel() override = default;

 private:
  struct CallBackArgs {
    int coll_id;
    int64_t src_actor_id;
    int64_t dst_actor_id;
    OfCollectiveBoxingActorContext *ctx;
    int64_t rank;
  };
  void VirtualKernelInit(KernelContext* ctx) override;
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
  void issueOfcclAllReduce(int64_t actor_id, int coll_id, const void* send_buff, void* recv_buff, KernelContext* ctx, const RankDesc& rank_desc) const;
};

void OfCollectiveBoxingGenericKernel::VirtualKernelInit(KernelContext* ctx) {
  const RankDesc& rank_desc = this->op_conf().of_collective_boxing_generic_conf().rank_desc();
  ctx->set_state(std::make_shared<OfCollectiveBoxingKernelState>(rank_desc));
}

void OfCollectiveBoxingGenericKernel::issueOfcclAllReduce(int64_t actor_id, int coll_id, const void* send_buff, void* recv_buff, KernelContext* ctx, const RankDesc& rank_desc) const {
  
  ofcclRankCtx_t ofccl_rank_ctx = dynamic_cast<OfCollectiveBoxingKernelState *>(ctx->state().get())->ofccl_rank_ctx();

  CallBackArgs *args = new CallBackArgs();
  // static成员可以在const函数中修改。
  args->dst_actor_id = actor_id; // 压进堆里的时候只能确定dst是给自己，只有真的执行的时候才知道是谁发送。
  args->coll_id = coll_id;
  args->ctx = GetOfCollectiveBoxingActorContext(ctx);
  args->rank = rank_desc.rank();

  // TODO(Panlichen): debug目的捕获this
  auto cb_lambda = [](int collIdFromCqe, void *args) {
    int64_t dst_actor_id = (static_cast<CallBackArgs *>(args))->dst_actor_id; // void不是类名，不能用dynamic
    Singleton<ActorMsgBus>::Get()->SendMsg(ActorMsg::BuildCollectiveMsg(dst_actor_id, dst_actor_id, CollectiveNegoCmd::kCollectiveDone));
    // VLOG(2) << "actor " << actor_id << " Rank<" << (static_cast<CallBackArgs *>(args))->rank << "> callback get cqe for coll_id = " << collIdFromCqe << " actor_ctx->coll_done_cnt_ = " << (static_cast<CallBackArgs *>(args))->ctx->coll_done_cnt_++;
    delete static_cast<CallBackArgs *>(args);
    return 0;
  };

  CallbackFunc cb_func = cb_lambda;

  // 目前的排序方法对于一个coll在一个iter里跑多次的情况不适用。但是有了计算图的话，应该不会一个coll在一个iter里跑多次。

  std::shared_ptr<OfRequestStore> of_request_store = Singleton<CollectiveMgr>::Get()->GetMutOfRequestStore();
  int64_t job_id = GetOfCollectiveBoxingActorContext(ctx)->task_proto().job_id();
  auto of_issue_params = std::make_shared<OfIssueParams>(
    coll_id, send_buff, recv_buff, rank_desc, args, cb_func, ofccl_rank_ctx, job_id,
    std::make_shared<HashMap<int64_t, int>>(of_request_store->job_id2curr_coll_id_vec),
    std::make_shared<HashMap<int64_t, std::vector<HashMap<int, int>>>>(of_request_store->job_id2local_coll_id2index)
  );

  // of_request_store->job_id2heap_mutex[job_id].lock();

  of_request_store->job_id2params_heap_in_one_iter[job_id].push(of_issue_params);

  VLOG(1) << "actor " << coll_id << " Rank<" << rank_desc.rank() << "> push coll into heap";

  // TODO(Panlichen): 会不会有多线程访问的问题？
  while (!of_request_store->job_id2params_heap_in_one_iter[job_id].empty()) {
    auto top_coll_params = of_request_store->job_id2params_heap_in_one_iter[job_id].top();
    int top_coll_id = top_coll_params->coll_id;

    // 这样的排序方法对于一个coll在一个iter里跑多次的情况不适用。但是有了计算图的话，应该不会一个coll在一个iter里跑多次。
    // 目前在resnet的场景下不会有问题。
    int curr_coll_id_vec = of_request_store->job_id2curr_coll_id_vec[job_id];
    auto curr_ordered_local_coll_ids = of_request_store->job_id2ordered_local_coll_ids[job_id][curr_coll_id_vec];

    int index_to_issue = of_request_store->job_id2index_to_issue[job_id];

    int next_coll_id_vec = (curr_coll_id_vec + 1) % of_request_store->NUM_COLL_ID_VEC;
    std::vector<int>& next_ordered_local_coll_ids = of_request_store->job_id2ordered_local_coll_ids[job_id][next_coll_id_vec];
    HashMap<int, int>& next_local_coll_id2index = of_request_store->job_id2local_coll_id2index[job_id][next_coll_id_vec];

    if (top_coll_id == curr_ordered_local_coll_ids[index_to_issue]) {
      // 可以发送的时候，同时更新next队列中的coll_id排列
      // TODO(Panlichen)
      // TODO(Panlichen)
      // TODO(Panlichen)
      // TODO(Panlichen)
      // TODO(Panlichen)
      // TODO(Panlichen): 在这里更新是不对的，应该在push那里更新；但是如果各个rank各自更新，又没意义了，我们最初的目的是希望各个rank顺序一致，要是确定一个打乱的顺序，就需要一个rank间的同步机制。
      next_ordered_local_coll_ids[index_to_issue] = top_coll_id;
      next_local_coll_id2index[top_coll_id] = index_to_issue;

      // 我来发送这个coll，也由我之后执行callback、发送msg
      ((CallBackArgs *)top_coll_params->cb_args)->src_actor_id = GetOfCollectiveBoxingActorContext(ctx)->actor_id();

      OF_NCCL_CHECK(ofcclRunAllReduce(top_coll_params->send_buff, top_coll_params->recv_buff, top_coll_params->coll_id, top_coll_params->cb_func, top_coll_params->cb_args, top_coll_params->ofccl_rank_ctx));
      of_request_store->job_id2params_heap_in_one_iter[job_id].pop();

      VLOG(1) << "actor " << coll_id << " Rank<" << rank_desc.rank() << "> issue coll_id = " << top_coll_id;

      of_request_store->job_id2index_to_issue[job_id] = (of_request_store->job_id2index_to_issue[job_id] + 1) % curr_ordered_local_coll_ids.size();
      // 切换到下一个coll_id vec
      if (of_request_store->job_id2index_to_issue[job_id] == 0) {
        of_request_store->job_id2curr_coll_id_vec[job_id] = (of_request_store->job_id2curr_coll_id_vec[job_id] + 1) % of_request_store->NUM_COLL_ID_VEC;
      }
    } else {
      // 当前堆顶的coll不能发射。
      break;
    }
  }

  // of_request_store->job_id2heap_mutex[job_id].unlock();
  
  // ！！！！！！！！！！为了记log加的逻辑！！！！！！！！
  // size_t count = 1;
  // const Shape shape = Shape(rank_desc.op_desc().shape());
  // FOR_RANGE(int, shape_ax, 0, shape.NumAxes()) { count *= shape.At(shape_ax); }
  // CHECK_GT(count, 0);
  // VLOG(2) << "actor " << actor_id << " Rank<" << rank_desc.rank() << "> ForwardDataContent invoke ofcclRunAllReduce with coll_id = " << coll_id  << " actor_ctx->coll_req_cnt_ = " << GetOfCollectiveBoxingActorContext(ctx)->coll_req_cnt_++; // << " send_buff = " << send_buff << " recv_buff = " << recv_buff;
}

void OfCollectiveBoxingGenericKernel::ForwardDataContent(KernelContext* ctx) const {
  VLOG(3) << "Enter OfCollectiveBoxingGenericKernel::ForwardDataContent";

  int64_t actor_id = GetOfCollectiveBoxingActorContext(ctx)->actor_id();

  // Singleton<ActorMsgBus>::Get()->SendMsg(ActorMsg::BuildCollectiveMsg(actor_id, actor_id, CollectiveNegoCmd::kCollectiveDone));
  // return;

  const RankDesc& rank_desc = this->op_conf().of_collective_boxing_generic_conf().rank_desc();
  
  // TODO(Panlichen): 目前只实现了AllReduce  
  if (rank_desc.op_desc().op_type() == kOpTypeAllReduce) {
    const void* send_buff = nullptr;
    void* recv_buff = nullptr;
    const DataType data_type = rank_desc.op_desc().data_type();
    if (GenericOpHasInput(rank_desc)) {
      const Blob* in = ctx->BnInOp2Blob("in");
      CHECK_EQ(in->data_type(), data_type);
      CHECK(in->shape() == ShapeView(GenericOpGetInputShape(rank_desc)));
      send_buff = in->dptr();
    }
    if (GenericOpHasOutput(rank_desc)) {
      Blob* out = ctx->BnInOp2Blob("out");
      CHECK_EQ(out->data_type(), data_type);
      CHECK(out->shape() == ShapeView(GenericOpGetOutputShape(rank_desc)));
      recv_buff = out->mut_dptr();
    }
    int coll_id = dynamic_cast<OfCollectiveBoxingKernelState *>(ctx->state().get())->coll_id();    
    issueOfcclAllReduce(actor_id, coll_id, send_buff, recv_buff, ctx, rank_desc);
  }
}

REGISTER_KERNEL(OperatorConf::kOfCollectiveBoxingGenericConf, OfCollectiveBoxingGenericKernel);

}  // namespace

}  // namespace oneflow
