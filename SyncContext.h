// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <memory>
// helium
#include "helium/TaskQueue.h"
// visionaray
#include "visionaray/detail/thread_pool.h"
// ours
#include "RenderingSemaphore.h"

namespace visionaray {

struct SyncContext
{
  typedef std::shared_ptr<SyncContext> SP;

  SyncContext();
  ~SyncContext();

  thread_pool threadPool;
#ifdef WITH_CUDA
  cudaStream_t renderingStream;
  cudaStream_t copyStream;
#elif defined(WITH_HIP)
  hipStream_t renderingStream;
  hipStream_t copyStream;
#else
  helium::tasking::TaskQueue taskQueue{64};
  RenderingSemaphore renderingSemaphore;
#endif
};

} // visionaray
