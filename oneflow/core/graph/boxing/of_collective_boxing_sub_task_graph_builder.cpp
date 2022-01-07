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
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/of_collective_boxing_reduce_task_node.h"
#include "oneflow/core/graph/of_collective_boxing_broadcast_task_node.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/collective_boxing_pack_task_node.h"
#include "oneflow/core/graph/collective_boxing_unpack_task_node.h"
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/job/nd_sbp_util.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

using namespace boxing::collective;

namespace {

void OfcclInitCollectiveNode(OfCollectiveBoxingReduceTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
  OperatorConf op_conf;
  op_conf.set_name(name);
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(DeviceType::kCUDA)));
  OfCollectiveBoxingReduceOpConf* conf = op_conf.mutable_of_collective_boxing_reduce_conf();
  *conf->mutable_lbi() = lbi;
  RankDesc* rank_desc = conf->mutable_rank_desc();
  OpDesc* op_desc = rank_desc->mutable_op_desc();
  op_desc->set_name(name);
  op_desc->set_op_type(op_type);
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeReduceScatter
      || op_type == OpType::kOpTypeReduce) {
    op_desc->set_reduce_method(ReduceMethod::kReduceMethodSum);
  }
  op_desc->set_data_type(logical_blob_desc.data_type());
  logical_blob_desc.shape().ToProto(op_desc->mutable_shape());
  op_desc->set_num_ranks(parallel_desc.parallel_num());
  if (op_type == OpType::kOpTypeBroadcast || op_type == OpType::kOpTypeReduce) {
    CHECK_GE(root, 0);
    CHECK_LT(root, parallel_desc.parallel_num());
    op_desc->set_root(root);
  } else {
    CHECK_EQ(root, -1);
  }
  op_desc->set_backend(Backend::kBackendOf);
  rank_desc->set_rank(parallel_id);

  const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
  const int64_t device_index = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
  const int64_t thrd_id = EncodeStreamIdToInt64(
      GenerateNamedTaskStreamId(machine_id, DeviceType::kCUDA, device_index, "OfCCL"));
  node->Init(machine_id, thrd_id, lbi, op_conf);
}

void OfcclInitCollectiveNode_Broadcast(OfCollectiveBoxingBroadcastTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
                                OperatorConf op_conf;
  // 这么写是肯定不行的，需要重构，重构的工作重点在于，1）把第一个参数变成一个通用性的；2）下面需要获取一个OpConf，这个OpConf也需要变成通用性的，但目前我们都没有。
  op_conf.set_name(name);
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(DeviceType::kCUDA)));
  OfCollectiveBoxingBroadcastOpConf* conf = op_conf.mutable_of_collective_boxing_broadcast_conf();
  *conf->mutable_lbi() = lbi;
  RankDesc* rank_desc = conf->mutable_rank_desc();
  OpDesc* op_desc = rank_desc->mutable_op_desc();
  op_desc->set_name(name);
  op_desc->set_op_type(op_type);
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeReduceScatter
      || op_type == OpType::kOpTypeReduce) {
    op_desc->set_reduce_method(ReduceMethod::kReduceMethodSum);
  }
  op_desc->set_data_type(logical_blob_desc.data_type());
  logical_blob_desc.shape().ToProto(op_desc->mutable_shape());
  op_desc->set_num_ranks(parallel_desc.parallel_num());
  if (op_type == OpType::kOpTypeBroadcast || op_type == OpType::kOpTypeReduce) {
    CHECK_GE(root, 0);
    CHECK_LT(root, parallel_desc.parallel_num());
    op_desc->set_root(root);
  } else {
    CHECK_EQ(root, -1);
  }
  op_desc->set_backend(Backend::kBackendOf);
  rank_desc->set_rank(parallel_id);

  const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
  const int64_t device_index = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
  const int64_t thrd_id = EncodeStreamIdToInt64(
      GenerateNamedTaskStreamId(machine_id, DeviceType::kCUDA, device_index, "OfCCL"));
  node->Init(machine_id, thrd_id, lbi, op_conf);
}

int64_t FindRootParallelId(const ParallelDesc& multi_device, const ParallelDesc& sole_device) {
  CHECK_EQ(sole_device.parallel_num(), 1);
  const int64_t root_machine_id = CHECK_JUST(sole_device.MachineId4ParallelId(0));
  const int64_t root_device_id = CHECK_JUST(sole_device.DeviceId4ParallelId(0));
  int64_t root_parallel_id = -1;
  FOR_RANGE(int64_t, i, 0, multi_device.parallel_num()) {
    if (CHECK_JUST(multi_device.MachineId4ParallelId(i)) == root_machine_id
        && CHECK_JUST(multi_device.DeviceId4ParallelId(i)) == root_device_id) {
      root_parallel_id = i;
      break;
    }
  }
  return root_parallel_id;
}

bool IsSourceTimeShape(const Shape& shape) { return shape.elem_cnt() == 1; }

class OfCollectiveBoxingReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingReduceSubTskGphBuilder);
  OfCollectiveBoxingReduceSubTskGphBuilder() = default;
  ~OfCollectiveBoxingReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
      const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (in_parallel_desc.parallel_num() > 1 && out_parallel_desc.parallel_num() == 1
        && in_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && in_sbp_parallel.has_partial_sum_parallel()) {
      const int64_t root_parallel_id = FindRootParallelId(in_parallel_desc, out_parallel_desc);
      if (root_parallel_id == -1) { return Error::BoxingNotSupportedError(); }

      const std::string op_name = "System-Boxing-OfCollectiveBoxingReduce-" + NewUniqueId();
      std::vector<OfCollectiveBoxingReduceTaskNode*> reduce_nodes;
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        auto* reduce_node = ctx->task_graph()->NewNode<OfCollectiveBoxingReduceTaskNode>();
        OfcclInitCollectiveNode(reduce_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduce, root_parallel_id);
        reduce_nodes.emplace_back(reduce_node);
      }
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        ctx->task_graph()->ConnectWithLbi(in_node, reduce_nodes.at(i), lbi);
        if(i == root_parallel_id){//end node, no sibling nodes
          sorted_out_tasks->emplace_back(reduce_nodes.at(i));
        }else {//other nodes, out edge to sibling
          int64_t next_node_index = (i + 1) % in_parallel_desc.parallel_num();
          TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            reduce_nodes.at(i), lbi, dynamic_cast<TaskNode*>(reduce_nodes.at(next_node_index))->MemZoneId121());
          ctx->task_graph()->ConnectWithLbi(proxy_node, reduce_nodes.at(next_node_index), lbi);
        }
      }
      return TRY(BuildSubTskGphBuilderStatus("OfCollectiveBoxingReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfCollectiveBoxingBroadcastSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingBroadcastSubTskGphBuilder);
  OfCollectiveBoxingBroadcastSubTskGphBuilder() = default;
  ~OfCollectiveBoxingBroadcastSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
      const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (in_parallel_desc.parallel_num() == 1 && out_parallel_desc.parallel_num() > 1
        && (in_parallel_desc.device_type() == DeviceType::kCUDA
            || (in_parallel_desc.device_type() == DeviceType::kCPU
                && logical_blob_desc.shape().elem_cnt() >= 1024))
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && out_sbp_parallel.has_broadcast_parallel()) {
      TaskNode* gpu_in_node = nullptr;
      int64_t root_parallel_id = -1;
      if (in_parallel_desc.device_type() == DeviceType::kCPU) {
        auto* cpu_in_node = sorted_in_tasks.front();
        root_parallel_id =
            SubTskGphBuilderUtil::FindNearestSrcParallelId(out_parallel_desc, in_parallel_desc, 0);
        gpu_in_node =
            ctx->task_graph()->GetProxyNode(cpu_in_node, lbi, out_parallel_desc, root_parallel_id);

      } else if (in_parallel_desc.device_type() == DeviceType::kCUDA) {
        root_parallel_id = FindRootParallelId(out_parallel_desc, in_parallel_desc);
        gpu_in_node = sorted_in_tasks.front();
      } else {
        return Error::BoxingNotSupportedError();
      }
      if (root_parallel_id == -1) { return Error::BoxingNotSupportedError(); }

      const std::string op_name = "System-Boxing-OfCollectiveBoxingBroadcast-" + NewUniqueId();
      std::vector<OfCollectiveBoxingBroadcastTaskNode*> broadcast_nodes;
      FOR_RANGE(int64_t, i, 0, out_parallel_desc.parallel_num()) {
        auto* broadcast_node = ctx->task_graph()->NewNode<OfCollectiveBoxingBroadcastTaskNode>();
        OfcclInitCollectiveNode_Broadcast(broadcast_node, out_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeBroadcast, root_parallel_id);
        broadcast_nodes.emplace_back(broadcast_node);
      }
      FOR_RANGE(int64_t, i, 0, out_parallel_desc.parallel_num()) {
        int64_t next_node_index = (i + 1) % out_parallel_desc.parallel_num();
        if (next_node_index != root_parallel_id) {// if the next node is root, it should not pass the data to the next node.
          if(i == root_parallel_id){
            ctx->task_graph()->ConnectWithLbi(gpu_in_node, broadcast_nodes.at(i), lbi);
          }
          TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            broadcast_nodes.at(i), lbi, dynamic_cast<TaskNode*>(broadcast_nodes.at(next_node_index))->MemZoneId121());
          ctx->task_graph()->ConnectWithLbi(proxy_node, broadcast_nodes.at(next_node_index), lbi);
        }
        sorted_out_tasks->emplace_back(broadcast_nodes.at(i));
      }
      return TRY(BuildSubTskGphBuilderStatus("OfCollectiveBoxingBroadcastSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfCollectiveBoxingAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingAllReduceSubTskGphBuilder);
  OfCollectiveBoxingAllReduceSubTskGphBuilder() = default;
  ~OfCollectiveBoxingAllReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
      const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (out_parallel_desc.Equals(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingP2B(in_sbp_parallel, out_sbp_parallel)) {
      const std::string op_name = "System-Boxing-OfCollectiveBoxingAllReduce-" + NewUniqueId();
            std::vector<OfCollectiveBoxingReduceTaskNode*> reduce_nodes;
      // the root id of the reduce part and broadcast part is the same.
      // the end of the reduce node has the full data and then broadcasts data
      const int64_t root_parallel_id = in_parallel_desc.parallel_num() - 1;
      // reduce part
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        auto* reduce_node = ctx->task_graph()->NewNode<OfCollectiveBoxingReduceTaskNode>();
        OfcclInitCollectiveNode(reduce_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduce, root_parallel_id);
        reduce_nodes.emplace_back(reduce_node);
      }
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        ctx->task_graph()->ConnectWithLbi(in_node, reduce_nodes.at(i), lbi);
        if(i != root_parallel_id){
          int64_t next_node_index = (i + 1) % in_parallel_desc.parallel_num();
          TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            reduce_nodes.at(i), lbi, dynamic_cast<TaskNode*>(reduce_nodes.at(next_node_index))->MemZoneId121());
          ctx->task_graph()->ConnectWithLbi(proxy_node, reduce_nodes.at(next_node_index), lbi);
        }
      }
      // reduce done
      // broadcast
      TaskNode* gpu_in_node = reduce_nodes.at(root_parallel_id);
            std::vector<OfCollectiveBoxingBroadcastTaskNode*> broadcast_nodes;
      FOR_RANGE(int64_t, i, 0, out_parallel_desc.parallel_num()) {
        auto* broadcast_node = ctx->task_graph()->NewNode<OfCollectiveBoxingBroadcastTaskNode>();
        OfcclInitCollectiveNode_Broadcast(broadcast_node, out_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeBroadcast, root_parallel_id);
        broadcast_nodes.emplace_back(broadcast_node);
      }
      FOR_RANGE(int64_t, i, 0, out_parallel_desc.parallel_num()) {
        int64_t next_node_index = (i + 1) % out_parallel_desc.parallel_num();
        if (next_node_index != root_parallel_id) {// if the next node is root, it should not pass the data to the next node.
          if(i == root_parallel_id){
            ctx->task_graph()->ConnectWithLbi(gpu_in_node, broadcast_nodes.at(i), lbi);
          }
          TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            broadcast_nodes.at(i), lbi, dynamic_cast<TaskNode*>(broadcast_nodes.at(next_node_index))->MemZoneId121());
          ctx->task_graph()->ConnectWithLbi(proxy_node, broadcast_nodes.at(next_node_index), lbi);
        }
        sorted_out_tasks->emplace_back(broadcast_nodes.at(i));
      }
      return TRY(BuildSubTskGphBuilderStatus("OfCollectiveBoxingAllReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

}  // namespace

OfCollectiveBoxingSubTskGphBuilder::OfCollectiveBoxingSubTskGphBuilder() {
  const CollectiveBoxingConf collective_boxing_conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OfCollectiveBoxingReduceSubTskGphBuilder());
  builders.emplace_back(new OfCollectiveBoxingBroadcastSubTskGphBuilder());
//   if (collective_boxing_conf.nccl_enable_all_to_all()) {
// #if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
//     builders.emplace_back(new NcclCollectiveBoxingAll2AllSubTskGphBuilder());
// #else
//     LOG(WARNING) << "nccl_enable_all_to_all is unavailable unless NCCL_VERSION > 2.7.0";
// #endif
//   }
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> OfCollectiveBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
    const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if (!GlobalJobDesc().Bool("__is_user_function__")) { return Error::BoxingNotSupportedError(); }
  if (!IsSourceTimeShape(time_shape)) { return Error::BoxingNotSupportedError(); }
  return chain_builder_->Build(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                               in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
                               in_sbp_parallel, out_sbp_parallel, time_shape);
}

}  // namespace oneflow
