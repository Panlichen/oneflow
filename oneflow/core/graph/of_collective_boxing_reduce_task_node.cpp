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
#include "oneflow/core/graph/of_collective_boxing_reduce_task_node.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

void OfCollectiveBoxingReduceTaskNode::Init(int64_t machine_id, int64_t thrd_id,
                                           const LogicalBlobId& lbi, const OperatorConf& op_conf) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_lbi(lbi);
  op_conf_ = op_conf;
}

void OfCollectiveBoxingReduceTaskNode::ProduceAllRegstsAndBindEdges() {
  // if (boxing::collective::GenericOpHasOutput( maybe this is different, all nodes have a out register
          // op_conf_.collective_boxing_generic_conf().rank_desc())) {
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
    this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); }); //怎么知道有几条edge
  // }
}

void OfCollectiveBoxingReduceTaskNode::ConsumeAllRegsts() {
  int64_t in_data_edge_cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    const auto order_it = edge2order_.find(edge);
    CHECK(order_it != edge2order_.end());
    ConsumeRegst("in_" + std::to_string(order_it->second), edge->GetSoleRegst());
    in_data_edge_cnt += 1;
  });
}

void OfCollectiveBoxingReduceTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_boxing_op = CHECK_JUST(ConstructOp(op_conf_));
  node->mut_op() = reduce_boxing_op;
  FOR_RANGE(size_t, i, 0, reduce_boxing_op->input_bns().size()) {
    const std::string& ibn = reduce_boxing_op->input_bns().Get(i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi());
  node->BindBnWithRegst(reduce_boxing_op->SoleObn(), out_regst);
  // what is this?
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(nullptr);
}

void OfCollectiveBoxingReduceTaskNode::InferProducedDataRegstTimeShape() {
  auto out_regst = GetProducedRegst("out");
  if (out_regst != nullptr) { out_regst->mut_data_regst_time_shape()->reset(new Shape({1, 1})); }
}

}  // namespace oneflow
