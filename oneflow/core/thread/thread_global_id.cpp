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
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

class GlobalIdStorage final {
 public:
  GlobalIdStorage() = default;
  ~GlobalIdStorage() = default;

  static GlobalIdStorage* Singleton() {
    static auto* storage = new GlobalIdStorage();
    return storage;
  }

  size_t Size() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return id2debug_string_.size();
  }

  Maybe<void> Emplace(int64_t id, const std::string& debug_string) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto& pair : id2debug_string_) { CHECK_NE_OR_RETURN(debug_string, pair.second); }
    CHECK_OR_RETURN(id2debug_string_.emplace(id, debug_string).second);
    return Maybe<void>::Ok();
  }

  bool IsInserted(int64_t id) {
    std::unique_lock<std::mutex> lock(mutex_);
    return id2debug_string_.count(id) > 0;
  }

  Maybe<void> TryEmplace(int64_t id, const std::string& debug_string) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto& pair : id2debug_string_) {
      if (pair.first == id) { CHECK_EQ_OR_RETURN(debug_string, pair.second); }
      if (pair.second == debug_string) { CHECK_EQ_OR_RETURN(id, pair.first); }
    }
    id2debug_string_.emplace(id, debug_string);
    return Maybe<void>::Ok();
  }

  Maybe<const std::string&> DebugString(int64_t id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return MapAt(id2debug_string_, id);
  }

  Maybe<void> Reset() {
    HashMap<int64_t, std::string>().swap(id2debug_string_);
    return Maybe<void>::Ok();
  }

 private:
  mutable std::mutex mutex_;
  HashMap<int64_t, std::string> id2debug_string_;
};

Optional<int64_t>* MutThreadLocalUniqueGlobalId() {
  static thread_local Optional<int64_t> global_id;
  return &global_id;
}

}  // namespace

size_t GetThreadGlobalIdCount() { return GlobalIdStorage::Singleton()->Size(); }

Maybe<void> CheckWorkerThreadThreadGlobalId(int64_t thread_global_id) {
  CHECK_GE_OR_RETURN(thread_global_id, 0) << "thread_global_id should be non negative";
  CHECK_LT_OR_RETURN(thread_global_id, TransportToken::MaxNumberOfThreadGlobalUId())
      << "thread_global_id should be less than " << TransportToken::MaxNumberOfThreadGlobalUId();
  CHECK_NE_OR_RETURN(thread_global_id, kThreadGlobalIdMain)
      << "thread_global_id " << thread_global_id << " has been used by main thread.";
  return Maybe<void>::Ok();
}

Maybe<void> InitThisThreadUniqueGlobalId(int64_t id, const std::string& debug_string) {
  JUST(GlobalIdStorage::Singleton()->Emplace(id, debug_string));
  auto* ptr = MutThreadLocalUniqueGlobalId();
  CHECK_OR_RETURN(!ptr->has_value());
  *ptr = id;
  return Maybe<void>::Ok();
}

const Optional<int64_t>& GetThisThreadGlobalId() { return *MutThreadLocalUniqueGlobalId(); }

Maybe<void> ResetThisThreadUniqueGlobalId() { return GlobalIdStorage::Singleton()->Reset(); }

ThreadGlobalIdGuard::ThreadGlobalIdGuard(int64_t thread_global_id)
    : old_thread_global_id_(GetThisThreadGlobalId()) {
  if (old_thread_global_id_.has_value()) {
    int64_t old_thread_global_id = CHECK_JUST(old_thread_global_id_);
    CHECK_EQ(old_thread_global_id, thread_global_id)
        << "nested ThreadGlobalIdGuard disabled. old thread_global_id: " << old_thread_global_id
        << ", new thread_global_id:" << thread_global_id;
  }
  *MutThreadLocalUniqueGlobalId() = thread_global_id;
}

ThreadGlobalIdGuard::~ThreadGlobalIdGuard() {
  *MutThreadLocalUniqueGlobalId() = old_thread_global_id_;
}

}  // namespace oneflow
