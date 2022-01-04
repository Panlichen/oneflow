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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

using namespace boxing::collective;

class OfCollectiveBoxingBroadcastOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingBroadcastOp);
  OfCollectiveBoxingBroadcastOp() = default;
  ~OfCollectiveBoxingBroadcastOp() override = default;

 private:
  Maybe<void> InitFromOpConf() override {
    CHECK(op_conf().has_of_collective_boxing_broadcast_conf());
    const RankDesc& rank_desc = op_conf().of_collective_boxing_broadcast_conf().rank_desc();
    if((rank_desc.rank() + 1) % rank_desc.op_desc().num_ranks()==rank_desc.op_desc().root()){
      EnrollRepeatedOutputBn("out", 1, false);
    }else{
      EnrollRepeatedOutputBn("out", 2, false);
    }
    
    EnrollInputBn("in", false);
    return Maybe<void>::Ok();
  }

  Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const {
    OperatorConf op_conf(this->op_conf());
    op_conf.set_name("undefined-op-name");
    CHECK(op_conf.has_of_collective_boxing_broadcast_conf());
    auto* boxing_conf = op_conf.mutable_of_collective_boxing_broadcast_conf();
    LogicalBlobId empty_logical_blob_id{};
    *boxing_conf->mutable_lbi() = empty_logical_blob_id;
    return SymbolOf(op_conf);
  }

  LogicalBlobId lbi4ibn(const std::string& input_bn) const override {
    return this->op_conf().of_collective_boxing_broadcast_conf().lbi();
  }

  LogicalBlobId lbi4obn(const std::string& output_bn) const override {
    return this->op_conf().of_collective_boxing_broadcast_conf().lbi();
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    UNIMPLEMENTED_THEN_RETURN();
  }

  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    const RankDesc& rank_desc = op_conf().of_collective_boxing_broadcast_conf().rank_desc();
    const DataType data_type = rank_desc.op_desc().data_type();
    BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_OR_RETURN(!in->is_dynamic());
    CHECK_EQ_OR_RETURN(in->data_type(), data_type);
    CHECK_EQ_OR_RETURN(in->shape(), Shape(rank_desc.op_desc().shape()));

    FOR_RANGE(int64_t, i, 1, output_bns().size()) {  
      BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
      out_i->set_data_type(data_type);
      out_i->mut_shape() = Shape(rank_desc.op_desc().shape());
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kOfCollectiveBoxingBroadcastConf, OfCollectiveBoxingBroadcastOp);

}  // namespace oneflow
