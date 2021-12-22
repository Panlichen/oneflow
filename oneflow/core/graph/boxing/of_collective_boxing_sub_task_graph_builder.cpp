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

void OcclInitCollectiveNode(CollectiveBoxingGenericTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
  OperatorConf op_conf;
  op_conf.set_name(name);
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(DeviceType::kCUDA)));
  CollectiveBoxingGenericOpConf* conf = op_conf.mutable_collective_boxing_generic_conf();
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
  op_desc->set_backend(Backend::kBackendNCCL);
  rank_desc->set_rank(parallel_id);

  const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
  const int64_t device_index = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
  const int64_t thrd_id = EncodeStreamIdToInt64(
      GenerateNamedTaskStreamId(machine_id, DeviceType::kCUDA, device_index, "NCCL"));
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

// class NcclCollectiveBoxingAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAllReduceSubTskGphBuilder);
//   NcclCollectiveBoxingAllReduceSubTskGphBuilder() = default;
//   ~NcclCollectiveBoxingAllReduceSubTskGphBuilder() override = default;

//   Maybe<SubTskGphBuilderStatus> Build(
//       SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
//       std::vector<TaskNode*>* sorted_out_tasks,
//       std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
//       const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
//       const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
//       const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
//     if (out_parallel_desc.Equals(in_parallel_desc)
//         && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
//         && out_parallel_desc.device_type() == DeviceType::kCUDA
//         && out_parallel_desc.parallel_num() > 1
//         && SubTskGphBuilderUtil::IsBoxingP2B(in_sbp_parallel, out_sbp_parallel)) {
//       const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllReduce-" + NewUniqueId();
//       FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
//         TaskNode* in_node = sorted_in_tasks.at(i);
//         auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
//         NcclInitCollectiveNode(collective_node, in_parallel_desc, i, op_name, lbi,
//                                logical_blob_desc, OpType::kOpTypeAllReduce, -1);
//         ctx->task_graph()->ConnectWithLbi(in_node, collective_node, lbi);
//         sorted_out_tasks->emplace_back(collective_node);
//       }
//       return TRY(BuildSubTskGphBuilderStatus("NcclCollectiveBoxingAllReduceSubTskGphBuilder", ""));
//     } else {
//       return Error::BoxingNotSupportedError();
//     }
//   }
// };

// class NcclCollectiveBoxingReduceScatterSubTskGphBuilder final : public SubTskGphBuilder {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingReduceScatterSubTskGphBuilder);
//   NcclCollectiveBoxingReduceScatterSubTskGphBuilder() = default;
//   ~NcclCollectiveBoxingReduceScatterSubTskGphBuilder() override = default;

//   Maybe<SubTskGphBuilderStatus> Build(
//       SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
//       std::vector<TaskNode*>* sorted_out_tasks,
//       std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
//       const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
//       const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
//       const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
//     if (out_parallel_desc.Equals(in_parallel_desc)
//         && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
//         && out_parallel_desc.device_type() == DeviceType::kCUDA
//         && out_parallel_desc.parallel_num() > 1
//         && logical_blob_desc.shape().At(0) % out_parallel_desc.parallel_num() == 0
//         && SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)
//         && out_sbp_parallel.split_parallel().axis() == 0) {
//       const std::string op_name =
//           "System-Boxing-NcclCollectiveBoxingReduceScatter-" + NewUniqueId();
//       FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
//         TaskNode* in_node = sorted_in_tasks.at(i);
//         auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
//         NcclInitCollectiveNode(collective_node, in_parallel_desc, i, op_name, lbi,
//                                logical_blob_desc, OpType::kOpTypeReduceScatter, -1);
//         ctx->task_graph()->ConnectWithLbi(in_node, collective_node, lbi);
//         sorted_out_tasks->emplace_back(collective_node);
//       }
//       return TRY(
//           BuildSubTskGphBuilderStatus("NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
//     } else {
//       return Error::BoxingNotSupportedError();
//     }
//   }
// };

// class NcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder);
//   NcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder() = default;
//   ~NcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder() override = default;

//   Maybe<SubTskGphBuilderStatus> Build(
//       SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
//       std::vector<TaskNode*>* sorted_out_tasks,
//       std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
//       const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
//       const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
//       const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
//     if (out_parallel_desc.Equals(in_parallel_desc)
//         && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
//         && out_parallel_desc.device_type() == DeviceType::kCUDA
//         && out_parallel_desc.parallel_num() > 1
//         && SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)
//         && logical_blob_desc.shape().At(out_sbp_parallel.split_parallel().axis())
//                    % out_parallel_desc.parallel_num()
//                == 0
//         && out_sbp_parallel.split_parallel().axis() != 0) {
//       const std::string op_name =
//           "System-Boxing-NcclCollectiveBoxingP2SNoncontinuous-" + NewUniqueId();
//       FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
//         const int64_t machine_id = CHECK_JUST(in_parallel_desc.MachineId4ParallelId(i));
//         const int64_t device_index = CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(i));
//         const int64_t thrd_id = EncodeStreamIdToInt64(
//             GenerateComputeTaskStreamId(machine_id, DeviceType::kCUDA, device_index));
//         TaskNode* in_node = sorted_in_tasks.at(i);
//         CollectiveBoxingPackTaskNode* pack_node =
//             ctx->task_graph()->NewNode<CollectiveBoxingPackTaskNode>();
//         pack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
//                         out_sbp_parallel, in_parallel_desc.parallel_num());
//         ctx->task_graph()->ConnectWithLbi(in_node, pack_node, lbi);

//         auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
//         NcclInitCollectiveNode(
//             collective_node, in_parallel_desc, i, op_name, lbi,
//             BlobDesc({logical_blob_desc.shape().elem_cnt()}, logical_blob_desc.data_type()),
//             OpType::kOpTypeReduceScatter, -1);
//         ctx->task_graph()->ConnectWithLbi(pack_node, collective_node, lbi);

//         CollectiveBoxingUnpackTaskNode* unpack_node =
//             ctx->task_graph()->NewNode<CollectiveBoxingUnpackTaskNode>();
//         unpack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
//                           out_sbp_parallel, in_parallel_desc.parallel_num());
//         ctx->task_graph()->ConnectWithLbi(collective_node, unpack_node, lbi);
//         sorted_out_tasks->emplace_back(unpack_node);
//       }
//       return TRY(
//           BuildSubTskGphBuilderStatus("NcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder", ""));
//     } else {
//       return Error::BoxingNotSupportedError();
//     }
//   }
// };

// class NcclCollectiveBoxingAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAllGatherSubTskGphBuilder);
//   NcclCollectiveBoxingAllGatherSubTskGphBuilder() = default;
//   ~NcclCollectiveBoxingAllGatherSubTskGphBuilder() override = default;

//   Maybe<SubTskGphBuilderStatus> Build(
//       SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
//       std::vector<TaskNode*>* sorted_out_tasks,
//       std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
//       const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
//       const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
//       const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
//     if (out_parallel_desc.EqualsIgnoringDeviceType(in_parallel_desc)
//         && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
//         && SubTskGphBuilderUtil::IsDeviceTypeCPUOrCUDA(in_parallel_desc)
//         && out_parallel_desc.device_type() == DeviceType::kCUDA
//         && out_parallel_desc.parallel_num() > 1
//         && logical_blob_desc.shape().At(0) % out_parallel_desc.parallel_num() == 0
//         && SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)
//         && in_sbp_parallel.split_parallel().axis() == 0) {
//       const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllGather-" + NewUniqueId();
//       FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
//         TaskNode* in_node = sorted_in_tasks.at(i);
//         TaskNode* in_node_proxy =
//             ctx->task_graph()->GetProxyNode(in_node, lbi, out_parallel_desc, i);
//         auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
//         NcclInitCollectiveNode(collective_node, out_parallel_desc, i, op_name, lbi,
//                                logical_blob_desc, OpType::kOpTypeAllGather, -1);
//         ctx->task_graph()->ConnectWithLbi(in_node_proxy, collective_node, lbi);
//         sorted_out_tasks->emplace_back(collective_node);
//       }
//       return TRY(BuildSubTskGphBuilderStatus("NcclCollectiveBoxingAllGatherSubTskGphBuilder", ""));
//     } else {
//       return Error::BoxingNotSupportedError();
//     }
//   }
// };

// class NcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder);
//   NcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder() = default;
//   ~NcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder() override = default;

//   Maybe<SubTskGphBuilderStatus> Build(
//       SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
//       std::vector<TaskNode*>* sorted_out_tasks,
//       std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
//       const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
//       const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
//       const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
//     if (out_parallel_desc.EqualsIgnoringDeviceType(in_parallel_desc)
//         && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
//         && SubTskGphBuilderUtil::IsDeviceTypeCPUOrCUDA(in_parallel_desc)
//         && out_parallel_desc.device_type() == DeviceType::kCUDA
//         && out_parallel_desc.parallel_num() > 1
//         && SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)
//         && logical_blob_desc.shape().At(in_sbp_parallel.split_parallel().axis())
//                    % out_parallel_desc.parallel_num()
//                == 0
//         && in_sbp_parallel.split_parallel().axis() != 0) {
//       const std::string op_name =
//           "System-Boxing-NcclCollectiveBoxingS2BNoncontinuous-" + NewUniqueId();
//       FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
//         const int64_t machine_id = CHECK_JUST(out_parallel_desc.MachineId4ParallelId(i));
//         const int64_t device_index = CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(i));
//         const int64_t thrd_id = EncodeStreamIdToInt64(
//             GenerateComputeTaskStreamId(machine_id, DeviceType::kCUDA, device_index));
//         TaskNode* in_node = sorted_in_tasks.at(i);
//         TaskNode* in_node_proxy =
//             ctx->task_graph()->GetProxyNode(in_node, lbi, out_parallel_desc, i);
//         CollectiveBoxingPackTaskNode* pack_node =
//             ctx->task_graph()->NewNode<CollectiveBoxingPackTaskNode>();
//         pack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
//                         out_sbp_parallel, in_parallel_desc.parallel_num());
//         ctx->task_graph()->ConnectWithLbi(in_node_proxy, pack_node, lbi);
//         auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
//         NcclInitCollectiveNode(
//             collective_node, out_parallel_desc, i, op_name, lbi,
//             BlobDesc({logical_blob_desc.shape().elem_cnt()}, logical_blob_desc.data_type()),
//             OpType::kOpTypeAllGather, -1);
//         ctx->task_graph()->ConnectWithLbi(pack_node, collective_node, lbi);
//         CollectiveBoxingUnpackTaskNode* unpack_node =
//             ctx->task_graph()->NewNode<CollectiveBoxingUnpackTaskNode>();
//         unpack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
//                           out_sbp_parallel, in_parallel_desc.parallel_num());
//         ctx->task_graph()->ConnectWithLbi(collective_node, unpack_node, lbi);
//         sorted_out_tasks->emplace_back(unpack_node);
//       }
//       return TRY(
//           BuildSubTskGphBuilderStatus("NcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder", ""));
//     } else {
//       return Error::BoxingNotSupportedError();
//     }
//   }
// };

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
      sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingReduceTaskNode>();
        OcclInitCollectiveNode(collective_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduce, root_parallel_id);
        ctx->task_graph()->ConnectWithLbi(in_node, collective_node, lbi);
        if (i == root_parallel_id) {
          sorted_out_tasks->emplace_back(collective_node);
        } else {
          sorted_ctrl_tasks->at(0).emplace_back(collective_node);
        }
      }
      return TRY(BuildSubTskGphBuilderStatus("OfCollectiveBoxingReduceSubTskGphBuilder", ""));
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
