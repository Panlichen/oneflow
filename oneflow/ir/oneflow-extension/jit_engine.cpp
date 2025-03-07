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
#include "OneFlow/Extension.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "OneFlow/kernel_launch/JITEngine.h"
#include "OneFlow/kernel_launch/RunContext.h"

extern "C" {
void* oneflow_okl_fetch_run_ctx(void* launcher, int64_t index) {
  return static_cast<typename std::tuple_element_t<0, oneflow::okl::FetchArgs>>(launcher)
      ->FetchRunCtx(static_cast<typename std::tuple_element_t<1, oneflow::okl::FetchArgs>>(index));
}

void* oneflow_okl_fetch_kernel(void* launcher, int64_t index) {
  return static_cast<typename std::tuple_element_t<0, oneflow::okl::FetchArgs>>(launcher)
      ->FetchKernel(static_cast<typename std::tuple_element_t<1, oneflow::okl::FetchArgs>>(index));
}

void oneflow_okl_launch(void* run_ctx, void* kernel) {
  const oneflow::user_op::OpKernel* engine =
      static_cast<typename std::tuple_element_t<1, oneflow::okl::LaunchArgs>>(kernel);

  oneflow::okl::RunContext* compute_ctx_ =
      static_cast<typename std::tuple_element_t<0, oneflow::okl::LaunchArgs>>(run_ctx);
  engine->Compute(compute_ctx_, compute_ctx_->FetchState(), compute_ctx_->FetchCache());
}
}  // extern "C"

namespace oneflow {
SharedLibs* MutSharedLibPaths() {
  static SharedLibs libs = {};
  return &libs;
}
const SharedLibs* SharedLibPaths() { return MutSharedLibPaths(); }
}  // namespace oneflow

oneflow::okl::JITEngine::JITEngine(mlir::ModuleOp module) {
  llvm::SmallVector<llvm::StringRef, 4> ext_libs(
      {oneflow::SharedLibPaths()->begin(), oneflow::SharedLibPaths()->end()});
  mlir::ExecutionEngineOptions jitOptions;
  jitOptions.transformer = {};
  jitOptions.jitCodeGenOptLevel = llvm::None;
  jitOptions.sharedLibPaths = ext_libs;

  auto jit_or_error = mlir::ExecutionEngine::create(module, jitOptions);
  CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                        << llvm::toString((jit_or_error).takeError());
  jit_or_error->swap(engine_);
}
