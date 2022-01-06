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
#include "oneflow/core/graph/of_collective_boxing_broadcast_task_node.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

void OfCollectiveBoxingBroadcastTaskNode::Init(int64_t machine_id, int64_t thrd_id,
                                           const LogicalBlobId& lbi, const OperatorConf& op_conf) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_lbi(lbi);
  op_conf_ = op_conf;
}

void OfCollectiveBoxingBroadcastTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", true);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
}

void OfCollectiveBoxingBroadcastTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    ConsumeRegst("in", edge->GetSoleRegst());
  });
}

void OfCollectiveBoxingBroadcastTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> broadcast_boxing_op = CHECK_JUST(ConstructOp(op_conf_));
  node->mut_op() = broadcast_boxing_op;
  const std::string& ibn = broadcast_boxing_op->input_bns().Get(0);
  node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi());
  node->BindBnWithRegst(broadcast_boxing_op->SoleObn(), out_regst);
  node->InferBlobDescs(nullptr);
}

void OfCollectiveBoxingBroadcastTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
