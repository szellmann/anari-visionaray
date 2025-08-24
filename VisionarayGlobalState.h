// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// helium
#include "helium/BaseGlobalDeviceState.h"
// visionaray
#include "visionaray/detail/thread_pool.h"
// ours
#include "DeviceObjectRegistry.h"
#include "RenderingSemaphore.h"

namespace visionaray {

struct Frame;

struct VisionarayGlobalState : public helium::BaseGlobalDeviceState
{
  thread_pool threadPool;

  struct ObjectUpdates
  {
    helium::TimeStamp lastBLSReconstructSceneRequest{0};
    helium::TimeStamp lastBLSCommitSceneRequest{0};
    helium::TimeStamp lastTLSReconstructSceneRequest{0};
  } objectUpdates;

  DeviceCopyableObjects dcos;
  DeviceObjectRegistry onDevice;

#ifdef WITH_CUDA
  cudaStream_t renderingStream;
#elif defined(WITH_HIP)
  hipStream_t renderingStream;
#else
  RenderingSemaphore renderingSemaphore;
#endif
  Frame *currentFrame{nullptr};

  anari::Device anariDevice{nullptr}; // public handle of _this_ helide instance

  // Helper methods //

  VisionarayGlobalState(ANARIDevice d);
  ~VisionarayGlobalState();
  void waitOnCurrentFrame() const;
};

// Helper functions/macros ////////////////////////////////////////////////////

inline VisionarayGlobalState *asVisionarayState(helium::BaseGlobalDeviceState *s)
{
  return (VisionarayGlobalState *)s;
}

#define VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISIONARAY_ANARI_TYPEFOR_DEFINITION(type)                              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // visionaray
