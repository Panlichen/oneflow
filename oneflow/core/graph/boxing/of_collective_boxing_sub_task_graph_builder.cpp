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
#include "oneflow/core/graph/of_collective_boxing_task_node.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/collective_boxing_pack_task_node.h"
#include "oneflow/core/graph/collective_boxing_unpack_task_node.h"
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/job/nd_sbp_util.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

using namespace boxing::of_collective;

namespace {

void OfcclInitCollectiveNode(OfCollectiveBoxingGenericTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
  OperatorConf op_conf;
  op_conf.set_name(name);
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(DeviceType::kCUDA)));
  OfCollectiveBoxingGenericOpConf* conf = op_conf.mutable_of_collective_boxing_generic_conf();
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
  op_desc->set_backend(Backend::kBackendOFCCL);
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

class OfcclCollectiveBoxingAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingAllReduceSubTskGphBuilder);
  OfcclCollectiveBoxingAllReduceSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingAllReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (out_parallel_desc.Equals(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingP2B(in_sbp_parallel, out_sbp_parallel)) {
      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingAllReduce-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllReduce, -1);
        ctx->task_graph()->ConnectWithLbi(in_node, of_collective_node, lbi);
        sorted_out_tasks->emplace_back(of_collective_node);
      }
      return TRY(BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingAllReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfcclCollectiveBoxingReduceScatterSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingReduceScatterSubTskGphBuilder);
  OfcclCollectiveBoxingReduceScatterSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingReduceScatterSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (out_parallel_desc.Equals(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1 && logical_blob_desc.shape().NumAxes() > 0
        && logical_blob_desc.shape().At(0) % out_parallel_desc.parallel_num() == 0
        && SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)
        && out_sbp_parallel.split_parallel().axis() == 0) {
      const std::string op_name =
          "System-Boxing-OfcclCollectiveBoxingReduceScatter-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduceScatter, -1);
        ctx->task_graph()->ConnectWithLbi(in_node, of_collective_node, lbi);
        sorted_out_tasks->emplace_back(of_collective_node);
      }
      return TRY(
          BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder);
  OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    const Shape& shape = logical_blob_desc.shape();
    const int64_t out_split_axis = out_sbp_parallel.split_parallel().axis();
    if (out_parallel_desc.Equals(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)
        && shape.NumAxes() > out_split_axis
        && shape.At(out_split_axis) % out_parallel_desc.parallel_num() == 0
        && out_sbp_parallel.split_parallel().axis() != 0) {
      const std::string op_name =
          "System-Boxing-OfcclCollectiveBoxingP2SNoncontinuous-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        const int64_t machine_id = CHECK_JUST(in_parallel_desc.MachineId4ParallelId(i));
        const int64_t device_index = CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(i));
        const int64_t thrd_id = EncodeStreamIdToInt64(
            GenerateComputeTaskStreamId(machine_id, DeviceType::kCUDA, device_index));
        TaskNode* in_node = sorted_in_tasks.at(i);
        CollectiveBoxingPackTaskNode* pack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingPackTaskNode>();
        pack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                        out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(in_node, pack_node, lbi);

        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(
            of_collective_node, in_parallel_desc, i, op_name, lbi,
            BlobDesc({logical_blob_desc.shape().elem_cnt()}, logical_blob_desc.data_type()),
            OpType::kOpTypeReduceScatter, -1);
        ctx->task_graph()->ConnectWithLbi(pack_node, of_collective_node, lbi);

        CollectiveBoxingUnpackTaskNode* unpack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingUnpackTaskNode>();
        unpack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                          out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(of_collective_node, unpack_node, lbi);
        sorted_out_tasks->emplace_back(unpack_node);
      }
      return TRY(
          BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfcclCollectiveBoxingAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingAllGatherSubTskGphBuilder);
  OfcclCollectiveBoxingAllGatherSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (out_parallel_desc.EqualsIgnoringDeviceType(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && SubTskGphBuilderUtil::IsDeviceTypeCPUOrCUDA(in_parallel_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1 && logical_blob_desc.shape().NumAxes() > 0
        && logical_blob_desc.shape().At(0) % out_parallel_desc.parallel_num() == 0
        && SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)
        && in_sbp_parallel.split_parallel().axis() == 0) {
      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingAllGather-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        TaskNode* in_node_proxy =
            ctx->task_graph()->GetProxyNode(in_node, lbi, out_parallel_desc, i);
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, out_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllGather, -1);
        ctx->task_graph()->ConnectWithLbi(in_node_proxy, of_collective_node, lbi);
        sorted_out_tasks->emplace_back(of_collective_node);
      }
      return TRY(BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingAllGatherSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder);
  OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    const Shape& shape = logical_blob_desc.shape();
    const int64_t in_split_axis = in_sbp_parallel.split_parallel().axis();
    if (out_parallel_desc.EqualsIgnoringDeviceType(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && SubTskGphBuilderUtil::IsDeviceTypeCPUOrCUDA(in_parallel_desc)
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)
        && shape.NumAxes() > in_split_axis && in_split_axis > 0
        && shape.At(in_split_axis) % out_parallel_desc.parallel_num() == 0) {
      const std::string op_name =
          "System-Boxing-OfcclCollectiveBoxingS2BNoncontinuous-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        const int64_t machine_id = CHECK_JUST(out_parallel_desc.MachineId4ParallelId(i));
        const int64_t device_index = CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(i));
        const int64_t thrd_id = EncodeStreamIdToInt64(
            GenerateComputeTaskStreamId(machine_id, DeviceType::kCUDA, device_index));
        TaskNode* in_node = sorted_in_tasks.at(i);
        TaskNode* in_node_proxy =
            ctx->task_graph()->GetProxyNode(in_node, lbi, out_parallel_desc, i);
        CollectiveBoxingPackTaskNode* pack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingPackTaskNode>();
        pack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                        out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(in_node_proxy, pack_node, lbi);
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(
            of_collective_node, out_parallel_desc, i, op_name, lbi,
            BlobDesc({logical_blob_desc.shape().elem_cnt()}, logical_blob_desc.data_type()),
            OpType::kOpTypeAllGather, -1);
        ctx->task_graph()->ConnectWithLbi(pack_node, of_collective_node, lbi);
        CollectiveBoxingUnpackTaskNode* unpack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingUnpackTaskNode>();
        unpack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                          out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(of_collective_node, unpack_node, lbi);
        sorted_out_tasks->emplace_back(unpack_node);
      }
      return TRY(
          BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};


class OfcclCollectiveBoxingReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingReduceSubTskGphBuilder);
  OfcclCollectiveBoxingReduceSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (in_parallel_desc.parallel_num() > 1 && out_parallel_desc.parallel_num() == 1
        && in_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && in_sbp_parallel.has_partial_sum_parallel()) {
      const int64_t root_parallel_id = FindRootParallelId(in_parallel_desc, out_parallel_desc);
      if (root_parallel_id == -1) { return Error::BoxingNotSupportedError(); }

      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingReduce-" + NewUniqueId();
      sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_in_tasks.at(i);
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, in_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduce, root_parallel_id);
        ctx->task_graph()->ConnectWithLbi(in_node, of_collective_node, lbi);
        if (i == root_parallel_id) {
          sorted_out_tasks->emplace_back(of_collective_node);
        } else {
          sorted_ctrl_tasks->at(0).emplace_back(of_collective_node);
        }
      }
      return TRY(BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder);
  OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder() = default;
  ~OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    if (in_parallel_desc.parallel_num() == 1 && out_parallel_desc.parallel_num() > 1
        && in_parallel_desc.device_type() == DeviceType::kCPU
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && logical_blob_desc.shape().elem_cnt() >= 1024
        && out_sbp_parallel.has_broadcast_parallel()
        // a potential optimization: flat the blob and then relax this requirement
        && logical_blob_desc.shape().NumAxes() > 0
        && logical_blob_desc.shape().At(0) % out_parallel_desc.parallel_num() == 0) {
      const TensorSliceView in_slice = GetBroadcastTensorSliceView(logical_blob_desc);
      SbpParallel split_sbp_parallel;
      split_sbp_parallel.mutable_split_parallel()->set_axis(0);
      std::vector<TensorSliceView> out_slices = GetTensorSliceView(
          out_parallel_desc.parallel_num(), split_sbp_parallel, logical_blob_desc);
      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingAllGather-" + NewUniqueId();
      FOR_RANGE(int64_t, out_id, 0, out_parallel_desc.parallel_num()) {
        const TensorSliceView& out_slice = out_slices.at(out_id);
        const int64_t nearest_in_parallel_id = SubTskGphBuilderUtil::FindNearestSrcParallelId(
            in_parallel_desc, out_parallel_desc, out_id);

        TaskNode* in_node = sorted_in_tasks.at(nearest_in_parallel_id);
        SliceBoxingTaskNode* slice_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
        // slice on cpu
        const auto in_machine_id = CHECK_JUST(in_parallel_desc.MachineId4ParallelId(0));
        int64_t thrd_id =
            EncodeStreamIdToInt64(GenerateComputeTaskStreamId(in_machine_id, DeviceType::kCPU, 0));
        slice_node->Init(lbi, out_slice, kSliceBoxingTaskModeCopy, in_machine_id, thrd_id);
        slice_node->ConnectToSrcNodeWithSlice(in_node, ctx->task_graph()->NewEdge(), in_slice);
        // copy to dst gpu
        TaskNode* slice_node_proxy =
            ctx->task_graph()->GetProxyNode(slice_node, lbi, out_parallel_desc, out_id);
        // allgather
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, out_parallel_desc, out_id, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllGather, -1);
        ctx->task_graph()->ConnectWithLbi(slice_node_proxy, of_collective_node, lbi);
        sorted_out_tasks->emplace_back(of_collective_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          "OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  };
};

class OfcclCollectiveBoxingBroadcastSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingBroadcastSubTskGphBuilder);
  OfcclCollectiveBoxingBroadcastSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingBroadcastSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
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

      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingBroadcast-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, out_parallel_desc.parallel_num()) {
        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, out_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeBroadcast, root_parallel_id);
        if (i == root_parallel_id) {
          ctx->task_graph()->ConnectWithLbi(gpu_in_node, of_collective_node, lbi);
        } else {
          std::string regst_desc_name;
          gpu_in_node->BuildCtrlRegstDesc(of_collective_node, &regst_desc_name);
          TaskEdge* edge = ctx->task_graph()->NewEdge();
          Connect<TaskNode>(gpu_in_node, edge, of_collective_node);
          gpu_in_node->BindEdgeWithProducedRegst(edge, regst_desc_name);
        }
        sorted_out_tasks->emplace_back(of_collective_node);
      }
      return TRY(BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingBroadcastSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class OfcclCollectiveBoxingAll2AllSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfcclCollectiveBoxingAll2AllSubTskGphBuilder);
  OfcclCollectiveBoxingAll2AllSubTskGphBuilder() = default;
  ~OfcclCollectiveBoxingAll2AllSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override {
    const Shape& shape = logical_blob_desc.shape();
    const int64_t in_split_axis = in_sbp_parallel.split_parallel().axis();
    const int64_t out_split_axis = out_sbp_parallel.split_parallel().axis();
    if (out_parallel_desc.EqualsIgnoringDeviceType(in_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && in_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.parallel_num() > 1
        && shape.NumAxes() > std::max(in_split_axis, out_split_axis)
        && shape.At(in_split_axis) % in_parallel_desc.parallel_num() == 0
        && shape.At(out_split_axis) % out_parallel_desc.parallel_num() == 0
        && in_sbp_parallel.split_parallel().axis() != out_sbp_parallel.split_parallel().axis()
        && SubTskGphBuilderUtil::IsBoxingS2S(in_sbp_parallel, out_sbp_parallel)) {
      const std::string op_name = "System-Boxing-OfcclCollectiveBoxingAll2All-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, in_parallel_desc.parallel_num()) {
        const int64_t machine_id = CHECK_JUST(in_parallel_desc.MachineId4ParallelId(i));
        const int64_t device_index = CHECK_JUST(in_parallel_desc.DeviceId4ParallelId(i));
        const int64_t thrd_id = EncodeStreamIdToInt64(
            GenerateComputeTaskStreamId(machine_id, DeviceType::kCUDA, device_index));
        TaskNode* in_node = sorted_in_tasks.at(i);
        CollectiveBoxingPackTaskNode* pack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingPackTaskNode>();
        pack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                        out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(in_node, pack_node, lbi);

        auto* of_collective_node = ctx->task_graph()->NewNode<OfCollectiveBoxingGenericTaskNode>();
        OfcclInitCollectiveNode(of_collective_node, out_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAll2All, -1);
        ctx->task_graph()->ConnectWithLbi(pack_node, of_collective_node, lbi);

        CollectiveBoxingUnpackTaskNode* unpack_node =
            ctx->task_graph()->NewNode<CollectiveBoxingUnpackTaskNode>();
        unpack_node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(), in_sbp_parallel,
                          out_sbp_parallel, in_parallel_desc.parallel_num());
        ctx->task_graph()->ConnectWithLbi(of_collective_node, unpack_node, lbi);
        sorted_out_tasks->emplace_back(unpack_node);
      }
      return TRY(BuildSubTskGphBuilderStatus("OfcclCollectiveBoxingAll2AllSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

}  // namespace

OfCollectiveBoxingSubTskGphBuilder::OfCollectiveBoxingSubTskGphBuilder() {
  // TODO(Panlichen):  maybe we need to extend this in resoure.proto
  const CollectiveBoxingConf collective_boxing_conf =
      Singleton<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OfcclCollectiveBoxingAllReduceSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingReduceScatterSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingP2SNoncontinuousSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingAllGatherSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingS2BNoncontinuousSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingReduceSubTskGphBuilder());
  builders.emplace_back(new OfCollectiveBoxingScatterThenOfcclAllGatherSubTskGphBuilder());
  builders.emplace_back(new OfcclCollectiveBoxingBroadcastSubTskGphBuilder());
  if (collective_boxing_conf.nccl_enable_all_to_all()) {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
    builders.emplace_back(new OfcclCollectiveBoxingAll2AllSubTskGphBuilder());
#else
    LOG(WARNING) << "nccl_enable_all_to_all is unavailable unless NCCL_VERSION > 2.7.0";
#endif
  }
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> OfCollectiveBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if (!GlobalJobDesc().Bool("__is_user_function__")) { return Error::BoxingNotSupportedError(); }

  VLOG(1) << "time_shape.elem_cnt() = " << time_shape.elem_cnt();
  // 为了让occl能真正接管pp的所有coll，关闭这个环境变量
  if (ParseBooleanFromEnv("ONEFLOW_TIME_SHAPE", false)) {
    VLOG(1) << "Judge IsSourceTimeShape";
    if (!IsSourceTimeShape(time_shape)) { return Error::BoxingNotSupportedError(); }
  } else {
    VLOG(1) << "Do not care about timeshape";
  }
  return chain_builder_->Build(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                               in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
                               in_sbp_parallel, out_sbp_parallel, time_shape);
}

}  // namespace oneflow
