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
#ifndef ONEFLOW_CORE_GRAPH_OF_COLLECTIVE_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_OF_COLLECTIVE_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/transport_task_node.h"

namespace oneflow {

class OfCollectiveBoxingGenericTaskNode : public TransportTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingGenericTaskNode);
  OfCollectiveBoxingGenericTaskNode() = default;
  ~OfCollectiveBoxingGenericTaskNode() override = default;

  void Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
            const OperatorConf& op_conf);

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;
  TaskType GetTaskType() const override { return TaskType::kOfCollectiveBoxingGeneric; }

  OperatorConf op_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OF_COLLECTIVE_BOXING_TASK_NODE_H_
