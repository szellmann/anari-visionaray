#pragma once

#include "DeviceCopyableObjects.h"
#include "VisionarayGlobalState.h"

namespace visionaray {

VSNRAY_FUNC inline
size_t linearIndex(const vec3i index, const vec3i dims)
{
  return index.z * size_t(dims.x) * dims.y
       + index.y * dims.x
       + index.x;
}

VSNRAY_FUNC inline
vec3i projectOnGrid(const vec3f V,
                    const vec3i dims,
                    const box3f worldBounds)
{
  const vec3f V01 = (V-worldBounds.min)/(worldBounds.max-worldBounds.min);
  return clamp(vec3i(V01*vec3f(dims)),vec3i(0),dims-vec3i(1));
}

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
  void init(int3 dims, box3 worldBounds);
  void cleanup();

  dco::GridAccel &visionarayAccel();

  box1 *valueRanges();

  void computeMaxOpacities(dco::TransferFunction tf);

  void dispatch(VisionarayGlobalState *s);
  void detach(VisionarayGlobalState *s);

private:

  dco::GridAccel vaccel;

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
