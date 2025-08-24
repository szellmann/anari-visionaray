// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceCopyableObjects.h"
#include "VisionarayGlobalState.h"
#include "GridAccel-common.h"

namespace visionaray {

VSNRAY_FUNC
inline void updateMC(const vec3i  mcID,
                     const vec3i  gridDims,
                     const float  value,
                     box1f       *valueRanges)
{
  // TODO: atomic or locked..
  valueRanges[linearIndex(mcID,gridDims)].min
      = min(valueRanges[linearIndex(mcID,gridDims)].min, value);
  valueRanges[linearIndex(mcID,gridDims)].max
      = max(valueRanges[linearIndex(mcID,gridDims)].max, value);
}

VSNRAY_FUNC
inline void updateMC(const vec3i  mcID,
                     const vec3i  gridDims,
                     const box1f  valueRange,
                     box1f       *valueRanges)
{
  // TODO: atomic or locked..
  valueRanges[linearIndex(mcID,gridDims)].min
      = min(valueRanges[linearIndex(mcID,gridDims)].min, valueRange.min);
  valueRanges[linearIndex(mcID,gridDims)].max
      = max(valueRanges[linearIndex(mcID,gridDims)].max, valueRange.max);
}

VSNRAY_FUNC
inline void updateMCStepSize(const vec3i  mcID,
                             const vec3i  gridDims,
                             const float  stepSize,
                             float       *stepSizes)
{
  // TODO: atomic or locked..
  stepSizes[linearIndex(mcID,gridDims)]
      = min(stepSizes[linearIndex(mcID,gridDims)], stepSize);
}

struct GridAccel
{
  GridAccel(VisionarayGlobalState *s);
  ~GridAccel();

  void init(int3 dims, box3 worldBounds);

  dco::GridAccel &visionarayAccel();

  bool isValid() const;

  VisionarayGlobalState *deviceState() const;

  void computeMaxOpacities(dco::TransferFunction1D tf);

private:

  dco::GridAccel vaccel;

  VisionarayGlobalState *m_state{nullptr};

  // step size to take per cell (e.g., for implicit ISOs)
  HostDeviceArray<float> m_stepSizes;

  // min/max ranges
  HostDeviceArray<box1> m_valueRanges;

  // majorants/max opacities
  HostDeviceArray<float> m_maxOpacities;

  // Number of MCs
  int3 m_dims{0};

  // World bounds the grid spans
  box3 m_worldBounds;

#ifdef WITH_CUDA
  cudaStream_t stream;
#elif defined(WITH_HIP)
  hipStream_t stream;
#endif
};

} // visionaray
