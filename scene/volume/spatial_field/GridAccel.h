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

struct GridAccel
{
  GridAccel(VisionarayGlobalState *s);
  void init(int3 dims, box3 worldBounds);
  void cleanup();

  dco::GridAccel &visionarayAccel();

  box1 *valueRanges();

  VisionarayGlobalState *deviceState() const;

  void computeMaxOpacities(dco::TransferFunction tf);

private:

  dco::GridAccel vaccel;

  VisionarayGlobalState *m_state{nullptr};

  // min/max ranges
  box1 *m_valueRanges{nullptr};

  // majorants/max opacities
  float *m_maxOpacities{nullptr};

  // Number of MCs
  int3 m_dims{0};

  // World bounds the grid spans
  box3 m_worldBounds;
};

} // visionaray
