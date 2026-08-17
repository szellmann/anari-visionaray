// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "VisionarayGlobalState.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), threadPool(std::thread::hardware_concurrency())
{
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamCreate(&renderingStream));
  CUDA_SAFE_CALL(cudaStreamCreate(&copyStream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamCreate(&renderingStream));
  HIP_SAFE_CALL(hipStreamCreate(&copyStream));
#endif
}

VisionarayGlobalState::~VisionarayGlobalState()
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
