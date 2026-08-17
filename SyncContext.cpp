// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#ifdef WITH_CUDA
// cuda
#include <cuda_runtime.h>
// visionaray
#include "visionaray/cuda/safe_call.h"
#elif defined(WITH_HIP)
// cuda
#include <hip/hip_runtime.h>
// visionaray
#include "visionaray/hip/safe_call.h"
#endif

#include "SyncContext.h"

namespace visionaray {

SyncContext::SyncContext()
    : threadPool(std::thread::hardware_concurrency())
{
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamCreate(&renderingStream));
  CUDA_SAFE_CALL(cudaStreamCreate(&copyStream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamCreate(&renderingStream));
  HIP_SAFE_CALL(hipStreamCreate(&copyStream));
#endif
}

SyncContext::~SyncContext()
{
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamDestroy(renderingStream));
  CUDA_SAFE_CALL(cudaStreamDestroy(copyStream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamDestroy(renderingStream));
  HIP_SAFE_CALL(hipStreamDestroy(copyStream));
#endif
}

} // namespace visionaray
