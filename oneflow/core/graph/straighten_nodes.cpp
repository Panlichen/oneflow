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
#include <memory>
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/straighten_nodes.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

namespace {

enum TaskClassifier : int {
  kWaitingOverlapNode = 0,
  kWaitingMainComputation = 1,
  kRunASAP = 2,
  kRunALAP = 3
};

// deciding parameter
// The sorting order of nodes for the straighten algorithm
enum StraightenOrder : int {
  kTributaryLayerAscend = 0,     // small tributary layers go first
  kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  kLayerAscend = 2,              // first in first out
  kMemoryIncrementAscend = 3,    // small memory increment go first
  kExceedTimeAscend = 4,         // small exceed time go first

  kTributaryLayerDescend = 100,     // large tributary layers go first
  kDistanceToOverlapDescend = 101,  // long distance to overlap go first
  kLayerDescend = 102,              // last in first out
  kMemoryIncrementDescend = 103,    // large memory increment go first
  kExceedTimeDescend = 104,         // large exceed time go first
};

// The difference between a descending order and its corresponding ascending order
const int kDiff4AscendDescend = 100;

class TopoStruct {
 public:
  TaskNode* node = nullptr;
  int32_t min_layer = -1;
  int32_t tributary_layer = -1;
  bool on_trunk = false;
  int32_t counter = 0;
  int32_t min_distance2overlap = -1;
  int64_t memory_increment = -1;
  TopoStruct* next_same_node = nullptr;
  int32_t exceed_time = -1;
  // We can have some other nodes in it for example
  // SbpNode<NdSbpSignature>* node;
  // SbpEdge<NdSbpSignature>* node;
  // Or we can omit all the pointers and leave all the useful parameters.

  // Drop down the tributary layer
  void DropTributaryLayer(int32_t upper_bound);

  void SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  void SpreadTrunk(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // The minimum computation distance from the beginning of this op to the next overlap node
  int32_t GetMinDistance2Overlap(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // Memory increment = (memory of out registers) - (memory of in registers)
  void ComputeMeomoryIncrement();

  // Exceed time = time of cpu - time of gpu
  // For most operators, the execution time on gpu exceed the execution time on cpu.
  // However, overlap is needed if time of cpu > time of gpu.
  void ComputeExceedTime();

  // TODO: We might design more deciding parameter and choose a right combination of them in the
  // future.

  // deciding parameter
  // kTributaryLayerAscend = 0,     // small tributary layers go first
  // kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  // kLayerAscend = 2,              // first in first out
  // kMemoryIncrementAscend = 3,    // small memory increment go first
  // kExceedTimeAscend = 4,         // small exceed time go first
  // kTributaryLayerDescend = 100,     // large tributary layers go first
  // kDistanceToOverlapDescend = 101,  // long distance to overlap go first
  // kLayerDescend = 102,              // last in first out
  // kMemoryIncrementDescend = 103,    // large memory increment go first
  // kExceedTimeDescend = 104,         // large exceed time go first
  int64_t GetDecidingParameter(StraightenOrder so) const;
};

StraightenAlgorithmTag sat;

// NOTE: Leave these code for debugging in the future
// static std::vector<StraightenOrder> decide_parameters({ParseIntegerFromEnv("Parameter0", 3),
//                                                        ParseIntegerFromEnv("Parameter1", 0),
//                                                        ParseIntegerFromEnv("Parameter2", 3)});
// The best parameter set for saving time is {102, 100}
// The best parameter set for saving memory is {3, 0}
static std::vector<StraightenOrder> decide_parameters;

// SAT, a.k.a. Scholastic Aptitude Test,
// is the college admission test in the United States of America.
void InitDecideParameters(StraightenAlgorithmTag sat) {
  decide_parameters.clear();
  if (sat == StraightenAlgorithmTag::kCompressMemory) {
    decide_parameters.push_back(StraightenOrder::kMemoryIncrementAscend);
    decide_parameters.push_back(StraightenOrder::kTributaryLayerAscend);
  } else if (sat == StraightenAlgorithmTag::kOverlap4Transfer) {
    decide_parameters.push_back(StraightenOrder::kLayerDescend);
    decide_parameters.push_back(StraightenOrder::kTributaryLayerDescend);
  } else {
    // sat==StraightenAlgorithmTag::kOverlap4CpuGpu
    decide_parameters.push_back(StraightenOrder::kExceedTimeDescend);
    decide_parameters.push_back(StraightenOrder::kLayerDescend);
    decide_parameters.push_back(StraightenOrder::kMemoryIncrementAscend);
  }
}

// move the head from source to target
void MoveFrontBetweenMaps(std::map<int32_t, TopoStruct*>& source,
                          std::map<int32_t, TopoStruct*>& target) {
  if (!source.empty()) {
    const auto& front = source.begin();
    target[front->first] = front->second;
    source.erase(front);
  }
};

bool ShouldRunASAP(TaskType task_type) {
  // They are sorted according to frequency of occurrences
  switch (task_type) {
    // We mark the number of occurrences in bert
    case TaskType::kDeviceTick:                  // 38
    case TaskType::kTick:                        // 8
    case TaskType::kSrcSubsetTick:               // 6
    case TaskType::kDstSubsetTick:               // 6
    case TaskType::kCriticalSectionWaitTick:     // 4
    case TaskType::kWaitAndSendIds:              // 2
    case TaskType::kPack:                        // 0
    case TaskType::kUnpack:                      // 0
    case TaskType::kRepeat:                      // 0
    case TaskType::kAcc:                         // 0
    case TaskType::kSourceTick:                  // 0
    case TaskType::kAccTick:                     // 0
    case TaskType::kAccCtrlTick:                 // ?
    case TaskType::kCase:                        // 0
    case TaskType::kEsac:                        // 0
    case TaskType::kReentrantLock: return true;  // 0
    default: return false;
  }
}

bool IsTransferNode(TaskType task_type) {
  // return task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64);
  // They are sorted according to frequency of occurrences
  switch (task_type) {
    // We mark the number of occurrences in bert
    case TaskType::kOfCollectiveBoxingGeneric:
    case TaskType::kCollectiveBoxingGeneric:        // 76
    case TaskType::kNcclSendRecvBoxing:             // ?
    case TaskType::kCopyHd:                         // 27
    case TaskType::kSliceBoxing:                    // 16
    case TaskType::kCopyCommNet:                    // 12
    case TaskType::kCollectiveBoxingPack:           // 8
    case TaskType::kCollectiveBoxingUnpack:         // 8
    case TaskType::kBoxingZeros:                    // 3
    case TaskType::kDistributeConcat:               // 0
    case TaskType::kDistributeSplit:                // 0
    case TaskType::kBoxingIdentity:                 // 0
    case TaskType::kDecodeH2D:                      // 0
    case TaskType::kSspVariableProxy: return true;  // 0
    default: return false;
  }
}

// Some operators have longer time in cpu and less time in gpu.
// Running those operators without overlap would cause large gap during each iteration.
// For example, expand dims would not execute any kernel on gpu but still need 10us to execute some
// functions on cpu.
bool ShortGpuTime(const OperatorConf& op_conf) {
  if (op_conf.has_variable_conf()) {
    // Variable operators would not be run. They just create tensors.
    // We do not visualize any execution in NVTX. (Even a tick operator has something in NVTX.)
    return true;
  }
  if (op_conf.has_user_conf()) {
    const auto& op_type_name = op_conf.user_conf().op_type_name();
    // They are sorted according to frequency of occurrences in stable diffusion
    if (op_type_name == "expand_dims"  // 90
        || op_type_name == "cast"      // 16
        || op_type_name == "expand"    // 2
    ) {
      return true;
    }
  }
  return false;
}

// Classifier for the set according to the task type
TaskClassifier GetTaskClassifier(const TaskNode* node) {
  // Check task.pb.h for detail
  // They are sorted according to frequency of judgement
  // frequency of judgement = the number of occurrences / the times of judgement
  TaskType task_type = node->GetTaskType();
  if (task_type == TaskType::kNormalForward) {
    const auto& op_conf = dynamic_cast<const CompTaskNode*>(node)->op()->op_conf();
    if (sat == StraightenAlgorithmTag::kOverlap4CpuGpu && ShortGpuTime(op_conf)) {
      return TaskClassifier::kWaitingOverlapNode;
    } else {
      return TaskClassifier::kWaitingMainComputation;
    }
  }
  if (IsTransferNode(task_type)) { return TaskClassifier::kWaitingOverlapNode; }
  if (task_type == TaskType::kCallbackNotify) { return TaskClassifier::kRunALAP; }
  if (ShouldRunASAP(task_type)) { return TaskClassifier::kRunASAP; }
  CHECK(false) << "Unclassified or invalid task type (" << task_type << ") showing up";
  // Throw a kRunASAP which means ignoring this node in the algorithm
  return TaskClassifier::kRunASAP;
}

// Drop down the maximum layer with the minimum layer from consumer
void TopoStruct::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < tributary_layer || tributary_layer < 0) { tributary_layer = upper_bound; }
}

// Should initialize the counter to be the number of out edges
// Compute maximum layer for tributaries
void TopoStruct::SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  if (counter || min_layer <= 0) { return; }
  int32_t producer_max_lay = 0;
  if (on_trunk) {
    producer_max_lay = min_layer - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = tributary_layer;
  }
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct->at(in);
    topo_struct_in.DropTributaryLayer(producer_max_lay);
    --topo_struct_in.counter;
    if (topo_struct_in.counter == 0) { topo_struct_in.SpreadTributaryLayer(task_node2topo_struct); }
  });
  // Reduce counter to -1 to avoid visiting again
  counter--;
}

// Judge if this node is on the trunk
// If so, judge it for its producer/upstream nodes
void TopoStruct::SpreadTrunk(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  // Skip it if this node is already judged.
  if (on_trunk) { return; }
  CHECK_GE(min_layer, 0) << "TopoStruct not initialized!";
  on_trunk = true;
  // If I am in the trunk, then all the children with (min_layer >= my layer id - 1) would be
  // considered as in the trunk
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct->at(in);
    if (topo_struct_in.min_layer == min_layer - 1) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
}

// The minimum computation distance from the beginning of this op to the next overlap
int32_t TopoStruct::GetMinDistance2Overlap(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  if (min_distance2overlap >= 0) { return min_distance2overlap; }
  // if this node should be overlapped by main computation nodes
  if (GetTaskClassifier(node) == TaskClassifier::kWaitingOverlapNode) {
    min_distance2overlap = 0;
    return min_distance2overlap;
  }
  // Otherwise, initialize it with a large number
  // Well, the total number in the task graph is large enough
  min_distance2overlap = task_node2topo_struct->size();
  node->ForEachNodeOnOutEdge([&](TaskNode* out) {
    min_distance2overlap =
        std::min(min_distance2overlap,
                 task_node2topo_struct->at(out).GetMinDistance2Overlap(task_node2topo_struct));
  });
  ++min_distance2overlap;
  return min_distance2overlap;
}

// Memory increment = (memory of out registers) - (memory of in registers)
void TopoStruct::ComputeMeomoryIncrement() {
  if (memory_increment < 0) {
    memory_increment = 0;
    for (const auto& produced_register : node->produced_regsts()) {
      if (produced_register.second->enable_reuse_mem()) {
        RegstDescProto temp_proto;
        produced_register.second->ToProto(&temp_proto);
        memory_increment += RtRegstDesc(temp_proto).TotalMainByteSize4AllRegst();
      }
    }
    for (const auto& consumed_register_list : node->consumed_regsts()) {
      for (const auto& consumed_register : consumed_register_list.second) {
        if (consumed_register->enable_reuse_mem()) {
          RegstDescProto temp_proto;
          consumed_register->ToProto(&temp_proto);
          memory_increment -= RtRegstDesc(temp_proto).TotalMainByteSize4AllRegst()
                              / consumed_register->consumers().size();
        }
      }
    }
  }
}

// Exceed time = time of cpu - time of gpu
void TopoStruct::ComputeExceedTime() {
  if (node->GetTaskType() == TaskType::kNormalForward
      && ShortGpuTime(dynamic_cast<const CompTaskNode*>(node)->op()->op_conf())) {
    exceed_time = 1;
  } else {
    exceed_time = 0;
  }
}

// deciding parameter
// kTributaryLayerAscend = 0,     // small tributary layers go first
// kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
// kLayerAscend = 2,              // first in first out
// kMemoryIncrementAscend = 3,    // small memory increment go first
// kExceedTimeAscend = 4,         // small exceed time go first
// kTributaryLayerDescend = 100,     // large tributary layers go first
// kDistanceToOverlapDescend = 101,  // long distance to overlap go first
// kLayerDescend = 102,              // last in first out
// kMemoryIncrementDescend = 103,    // large memory increment go first
// kExceedTimeDescend = 104,         // large exceed time go first
int64_t TopoStruct::GetDecidingParameter(StraightenOrder so) const {
  int64_t sign = 1;
  if (so >= kDiff4AscendDescend) {
    so = StraightenOrder(int(so) - kDiff4AscendDescend);
    sign = -1;
  }
  switch (so) {
    case StraightenOrder::kTributaryLayerAscend: return sign * tributary_layer;
    case StraightenOrder::kDistanceToOverlapAscend: return sign * min_distance2overlap;
    case StraightenOrder::kLayerAscend: return sign * min_layer;
    case StraightenOrder::kMemoryIncrementAscend: return sign * memory_increment;
    case StraightenOrder::kExceedTimeAscend: return sign * exceed_time;
    default: return 0;
  }
}

// Find the trunk of the task graph, then reduce the wait time for tributaries
void FindTrunk(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  // Find the maximum layer number
  int32_t max_min_layer = -1;
  for (const auto& pair : *task_node2topo_struct) {
    if (max_min_layer < pair.second.min_layer) { max_min_layer = pair.second.min_layer; }
  }
  // All the nodes with min_layer>=trunk_end_id would be considered as trunk nodes
  // The last 5 layers would be considered as in trunk anyway.
  int32_t trunk_end_id = max_min_layer - 4;
  for (auto& pair : *task_node2topo_struct) {
    auto& topo_struct = pair.second;
    // Initialize the counter and Tributary Layer
    topo_struct.counter = pair.first->out_edges().size();
    topo_struct.tributary_layer = max_min_layer;
    // Find out all the nodes on the trunk.
    if (topo_struct.min_layer >= trunk_end_id) { topo_struct.SpreadTrunk(task_node2topo_struct); }
  }

  for (auto& pair : *task_node2topo_struct) {
    // Compute maximum layer for tributaries
    pair.second.SpreadTributaryLayer(task_node2topo_struct);
    // Set the min_distance2overlap for each topological structure
    pair.second.GetMinDistance2Overlap(task_node2topo_struct);
  }
}

void UpdateSat(const HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  sat = GlobalJobDesc().job_conf().straighten_algorithm_tag_in_task_graph();
  if (sat == StraightenAlgorithmTag::kOverlap4CpuGpu) {
    // If not cpu nodes, then the overlap strategy between cpu and gpu might consume large memory
    bool exist_cpu_nodes = false;
    for (const auto& pair : task_node2topo_struct) {
      // Found a cpu node
      if (pair.second.exceed_time == 1) {
        exist_cpu_nodes = true;
        break;
      }
    }
    if (!exist_cpu_nodes) {
      // Switch to the compress memory strategy, the default one
      // Since the overlap strategy for transfer might not be working on 1n1d.
      sat = StraightenAlgorithmTag::kCompressMemory;
    }
  }
}

}  // anonymous namespace

void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes) {
  // The function for settle the order in the graph
  int64_t order_in_graph = 0;

  // Generate topological data structure for each task node
  HashMap<TaskNode*, TopoStruct> task_node2topo_struct;
  // Determine the same nodes which should run simultaneously
  HashMap<int32_t, HashMap<int32_t, std::map<int32_t, TopoStruct*>>>
      task_type2machine_id2node_id2topo_structs;
  std::map<int32_t, TopoStruct*> min_node_id2topo_struct;
  int32_t previous_min_layer = 0;
  task_graph->TopoForEachNode([&](TaskNode* node) {
    auto& topo_struct = task_node2topo_struct[node];
    topo_struct.node = node;
    topo_struct.ComputeMeomoryIncrement();
    topo_struct.ComputeExceedTime();
    if (node->in_edges().empty()) {
      topo_struct.min_layer = 0;
    } else {
      int32_t max_min_layer = 0;
      node->ForEachNodeOnInEdge([&](TaskNode* in) {
        max_min_layer = std::max(max_min_layer, task_node2topo_struct[in].min_layer);
      });
      topo_struct.min_layer = max_min_layer + 1;
      // Deal with all the nodes with min_layer=previous_min_layer
      if (max_min_layer >= previous_min_layer) {
        // Using "7" to represent "and"
        // a7b means a pair (a, b)
        for (auto& task_type7machine_id2node_id2topo_structs :
             task_type2machine_id2node_id2topo_structs) {
          auto& machine_id2node_id2topo_structs = task_type7machine_id2node_id2topo_structs.second;
          // Initializing the smallest node id for each machine
          for (auto& machine_id7node_id2topo_structs : machine_id2node_id2topo_structs) {
            MoveFrontBetweenMaps(machine_id7node_id2topo_structs.second, min_node_id2topo_struct);
          }

          while (!min_node_id2topo_struct.empty()) {
            // auto* topo_struct_min_node_id = min_node_id2topo_struct.begin()->second;
            // Store the same nodes in different machines
            std::vector<TopoStruct*> same_nodes;
            for (auto& min_node_id7topo_struct : min_node_id2topo_struct) {
              auto* curr_topo_struct = min_node_id7topo_struct.second;
              // Find out all the same nodes
              // Stop using Visual string before we find a better key
              // Currently we can use the topological structure and node id to decide the same nodes
              same_nodes.push_back(curr_topo_struct);
            }
            // Cyclize them
            for (int32_t i = 1; i < same_nodes.size(); i++) {
              same_nodes[i - 1]->next_same_node = same_nodes[i];
            }
            (*same_nodes.rbegin())->next_same_node = same_nodes[0];
            // Delete them and add new candidates
            for (auto* same_node_topo_struct : same_nodes) {
              // Erase them from min_node_id2topo_struct
              min_node_id2topo_struct.erase(same_node_topo_struct->node->node_id());
              // Add new candidate
              MoveFrontBetweenMaps(
                  machine_id2node_id2topo_structs[same_node_topo_struct->node->machine_id()],
                  min_node_id2topo_struct);
            }
          }
        }
        // Renew the previous min_layer at the end
        previous_min_layer = topo_struct.min_layer;
      }
    }
    // Put the topo structure into the map, waiting for determine the same nodes
    task_type2machine_id2node_id2topo_structs[node->GetTaskType()][node->machine_id()]
                                             [node->node_id()] = &topo_struct;
  });

  // Generate other parameters in the topological data structure
  FindTrunk(&task_node2topo_struct);

  // Update sat, since sat might be changed in previous jobs
  UpdateSat(task_node2topo_struct);
  // Decide which node should run first
  InitDecideParameters(sat);
  VLOG(3) << "Straightening order: ";
  for (int32_t decide_parameter : decide_parameters) { VLOG(3) << decide_parameter; }

  // Order in the waiting sets
  struct comp {
    bool operator()(const TopoStruct* a, const TopoStruct* b) const {
      for (auto decide_parameter : decide_parameters) {
        auto decide_parameter_a = a->GetDecidingParameter(decide_parameter);
        auto decide_parameter_b = b->GetDecidingParameter(decide_parameter);
        if (decide_parameter_a != decide_parameter_b) {
          return decide_parameter_a < decide_parameter_b;
        }
      }
      return a->node->node_id() < b->node->node_id();
    }
  };

  // Classify sets for the task nodes
  // 0, TaskClassifier::kWaitingOverlapNode
  // It contains transfer nodes, and those with less time in gpu if request.
  // std::set<TopoStruct*, comp> waiting_overlap_node;
  // 1, TaskClassifier::kWaitingMainComputation
  // std::set<TopoStruct*, comp> waiting_main_computation;
  // 2, TaskClassifier::kRunASAP , run as soon as possible
  // std::set<TopoStruct*, comp> run_asap;
  // 3, TaskClassifier::kRunALAP , run as late as possible
  // std::set<TopoStruct*, comp> run_alap;
  const int32_t num_classifier = 4;
  std::vector<std::set<TopoStruct*, comp>> waiting_lists(num_classifier);

  std::vector<int32_t> remain_task_nums(num_classifier, 0);

  auto SetOrderInGraph = [&](TaskNode* task_node) {
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes->emplace_back(task_node);
    ++order_in_graph;
  };

  // wait in the list
  auto wait = [&](TaskNode* node) {
    TopoStruct* first_topo_struct = &task_node2topo_struct[node];
    // Check if all the same nodes are ready simultaneously
    TopoStruct* curr_topo_struct = first_topo_struct->next_same_node;
    while (curr_topo_struct && curr_topo_struct != first_topo_struct) {
      if (curr_topo_struct->counter) { return; }
      curr_topo_struct = curr_topo_struct->next_same_node;
    }
    // Add all the same nodes at the same time
    curr_topo_struct = first_topo_struct;
    auto& waiting_list = waiting_lists[GetTaskClassifier(node)];
    while (true) {
      waiting_list.insert(curr_topo_struct);
      // Reduce counter then this node will never be added again
      // Though inserting into a map twice does not matter because of the same keys
      curr_topo_struct->counter--;
      curr_topo_struct = curr_topo_struct->next_same_node;
      if ((!curr_topo_struct) || (curr_topo_struct == first_topo_struct)) { break; }
    }
  };

  // initialization
  task_graph->ForEachNode([&](TaskNode* node) {
    int32_t count = node->in_edges().size();
    task_node2topo_struct[node].counter = count;
    if (count == 0) { wait(node); }
    remain_task_nums[GetTaskClassifier(node)]++;
  });

  // Finish execution
  auto finish_execution = [&](TaskNode* node) {
    node->ForEachNodeOnOutEdge([&](TaskNode* out) {
      --(task_node2topo_struct[out].counter);
      if (task_node2topo_struct[out].counter == 0) { wait(out); }
    });
  };

  // Move the first node of the waiting list to the execution list
  auto move2execution_list = [&](std::set<TopoStruct*, comp>& waiting_list,
                                 std::vector<TaskNode*>& execution_list) {
    TaskNode* first_node = (*waiting_list.begin())->node;
    int32_t execution_num = 0;
    TopoStruct* first_topo_struct = &task_node2topo_struct[first_node];
    // Find all the same nodes in different machine
    // They should be run simultaneously
    TopoStruct* curr_topo_struct = first_topo_struct;
    while (true) {
      execution_num++;
      execution_list.push_back(curr_topo_struct->node);
      waiting_list.erase(curr_topo_struct);
      // move and maybe leave
      curr_topo_struct = curr_topo_struct->next_same_node;
      if ((!curr_topo_struct) || (curr_topo_struct == first_topo_struct)) { break; }
    }
    CHECK_GT(execution_num, 0) << "Error, no task nodes are moved to the execution list";
  };

  // Execute the first n nodes in the waiting list
  auto execute = [&](int32_t list_classifier, int32_t n, bool if_reverse = false) {
    // n > 0
    if (n <= 0) { return; }
    auto& waiting_list = waiting_lists[list_classifier];
    std::vector<TaskNode*> execution_list;
    int32_t count = 0;
    // Move to the execution list
    while (!waiting_list.empty()) {
      move2execution_list(waiting_list, execution_list);
      count++;
      if (count >= n) { break; }
    }
    remain_task_nums[list_classifier] -= execution_list.size();
    // Set the order and then remove from the execution list
    for (auto* node : execution_list) {
      SetOrderInGraph(node);
      finish_execution(node);
    }
  };

  // straightening
  while (true) {
    if (waiting_lists[TaskClassifier::kRunASAP].empty()) {
      if (waiting_lists[TaskClassifier::kWaitingOverlapNode].empty()) {
        if (waiting_lists[TaskClassifier::kWaitingMainComputation].empty()) {
          if (waiting_lists[TaskClassifier::kRunALAP].empty()) {
            // All the waiting lists are empty
            break;
          } else {
            // Execute all the nodes left
            execute(TaskClassifier::kRunALAP, waiting_lists[TaskClassifier::kRunALAP].size());
          }
        } else {
          // Execute one computation node
          execute(TaskClassifier::kWaitingMainComputation, 1);
        }
      } else {
        int32_t computation_num =
            std::min(int32_t(waiting_lists[TaskClassifier::kWaitingMainComputation].size()
                             / (waiting_lists[TaskClassifier::kWaitingOverlapNode].size())),
                     remain_task_nums[TaskClassifier::kWaitingMainComputation]
                         / remain_task_nums[TaskClassifier::kWaitingOverlapNode]);
        // Holding the node to be overlapped
        std::vector<TaskNode*> overlap_execution_list;
        move2execution_list(waiting_lists[TaskClassifier::kWaitingOverlapNode],
                            overlap_execution_list);
        remain_task_nums[TaskClassifier::kWaitingOverlapNode] -= overlap_execution_list.size();
        for (auto* overlap_node : overlap_execution_list) { SetOrderInGraph(overlap_node); }
        // Overlap the node with computation from the trunk
        execute(TaskClassifier::kWaitingMainComputation, computation_num);

        // Release the overlap node
        for (auto* overlap_node : overlap_execution_list) { finish_execution(overlap_node); }
      }
    } else {
      execute(TaskClassifier::kRunASAP, waiting_lists[TaskClassifier::kRunASAP].size());
    }
  }
}

}  // namespace oneflow
