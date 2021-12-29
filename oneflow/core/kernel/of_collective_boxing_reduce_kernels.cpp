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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

class OfCollectiveBoxingReduceKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingReduceKernel);
  OfCollectiveBoxingReduceKernel() = default;
  ~OfCollectiveBoxingReduceKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
};

void OfCollectiveBoxingReduceKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  std::unique_ptr<ep::primitive::Add> primitive =
      ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->stream()->device_type(),
                                                             out->data_type());
  CHECK(primitive);
  if (this->op_attribute().input_bns().size() == 1){ //start node
    const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", 0));
    AutoMemcpy(ctx->stream(), out, in_i);
  } else {
    FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
      const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
      primitive->Launch(ctx->stream(), out->dptr(), in_i->dptr(), out->mut_dptr(),
                          out->shape().elem_cnt());
    }
  }
}

REGISTER_KERNEL(OperatorConf::kOfCollectiveBoxingReduceConf, OfCollectiveBoxingReduceKernel);

}  // namespace oneflow
