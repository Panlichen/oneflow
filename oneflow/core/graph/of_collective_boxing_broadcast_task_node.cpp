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
  int64_t out_data_edge_cnt = 0;
  ForEachOutDataEdge([&](TaskEdge* edge) {
    edge->AddRegst("out_" + std::to_string(out_data_edge_cnt), ProduceRegst("out_" + std::to_string(out_data_edge_cnt), false, 1, 1));
    out_data_edge_cnt += 1;
  });
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
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  in_regst->AddLbi(lbi());
  node->BindBnsWithRegst(broadcast_boxing_op->SoleObn(), in_regst);
  FOR_RANGE(size_t, i, 0, broadcast_boxing_op->output_bns().size()) {
    const std::string& obn = broadcast_boxing_op->output_bns().Get(i);
    node->BindBnWithRegst(obn, GetProducedRegst("out_" + std::to_string(i)));
  }
  node->InferBlobDescs(nullptr);
}

void OfCollectiveBoxingBroadcastTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
