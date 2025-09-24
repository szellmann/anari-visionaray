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


VSNRAY_FUNC
inline void rasterizeBox(dco::GridAccel accel,
                         const box3f &box,
                         const box1f &valueRange,
                         const float stepSize)
{
  const vec3i loMC = projectOnGrid(box.min,accel.dims,accel.worldBounds);
  const vec3i upMC = projectOnGrid(box.max,accel.dims,accel.worldBounds);

  for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
    for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
      for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
        const vec3i mcID(mcx,mcy,mcz);
        updateMC(mcID,accel.dims,valueRange,accel.valueRanges);
        updateMCStepSize(mcID,accel.dims,stepSize,accel.stepSizes);
      }
    }
  }
}
struct GridAccel
{
  GridAccel(VisionarayGlobalState *s);
  ~GridAccel();

  void init(int3 dims, box3 worldBounds, box3 gridBounds);

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
