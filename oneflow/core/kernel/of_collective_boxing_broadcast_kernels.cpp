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

class OfCollectiveBoxingBroadcastKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingBroadcastKernel);
  OfCollectiveBoxingBroadcastKernel() = default;
  ~OfCollectiveBoxingBroadcastKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
};

void OfCollectiveBoxingBroadcastKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* in = ctx->BnInOp2Blob("in");
  Blob* out = ctx->BnInOp2Blob("out");
  AutoMemcpy(ctx->stream(), out, in);
  // if (this->op_attribute().output_bns().size() == 1){
  //   Blob* out = ctx->BnInOp2Blob(GenRepeatedBn("out", 0));
  //   AutoMemcpy(ctx->stream(), out, in);
  // } else {
  //   FOR_RANGE(int64_t, i, 0, this->op_attribute().output_bns().size()) {
  //     Blob* out_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
  //     AutoMemcpy(ctx->stream(), out_i, in);
  //   }
  // }
}

REGISTER_KERNEL(OperatorConf::kOfCollectiveBoxingBroadcastConf, OfCollectiveBoxingBroadcastKernel);

}  // namespace oneflow
