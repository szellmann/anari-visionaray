// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <common.h>

namespace visionaray {
namespace dco {

  struct UElemGrid
  {
    uint32_t gridID;
    int3 dims;
    aabb domain;
    uint64_t scalarsOffset;
    float *scalarsBuffer;
  };

  VSNRAY_FUNC
  inline aabb get_bounds(const UElemGrid &grid)
  {
    return grid.domain;
  }

  inline void split_primitive(
      aabb &L, aabb &R, float plane, int axis, const UElemGrid &grid)
  {
    assert(0);
  }

  VSNRAY_FUNC
  inline bool intersectGrid(const UElemGrid &grid, const float3 P, float& retVal)
  {
    if (!grid.domain.contains(P))
      return false;

    int3 dims = grid.dims;
    int3 numScalars = dims+int3(1);
    float3 cellSize = grid.domain.size()/float3(dims);
    float3 objPos = (P-grid.domain.min)/cellSize;
    int3 imin(objPos);
    int3 imax = min(imin+int3(1),numScalars-int3(1));

    auto linearIndex = [numScalars](const int x, const int y, const int z) {
                         return z*numScalars.y*numScalars.x + y*numScalars.x + x;
                       };

    const float *scalars = grid.scalarsBuffer + grid.scalarsOffset;

    float f1 = scalars[linearIndex(imin.x,imin.y,imin.z)];
    float f2 = scalars[linearIndex(imax.x,imin.y,imin.z)];
    float f3 = scalars[linearIndex(imin.x,imax.y,imin.z)];
    float f4 = scalars[linearIndex(imax.x,imax.y,imin.z)];

    float f5 = scalars[linearIndex(imin.x,imin.y,imax.z)];
    float f6 = scalars[linearIndex(imax.x,imin.y,imax.z)];
    float f7 = scalars[linearIndex(imin.x,imax.y,imax.z)];
    float f8 = scalars[linearIndex(imax.x,imax.y,imax.z)];

#define EMPTY(x) isnan(x)
    if (EMPTY(f1) || EMPTY(f2) || EMPTY(f3) || EMPTY(f4) ||
        EMPTY(f5) || EMPTY(f6) || EMPTY(f7) || EMPTY(f8))
      return false;

    float3 frac = objPos-float3(imin);

    float f12 = lerp_r(f1,f2,frac.x);
    float f56 = lerp_r(f5,f6,frac.x);
    float f34 = lerp_r(f3,f4,frac.x);
    float f78 = lerp_r(f7,f8,frac.x);

    float f1234 = lerp_r(f12,f34,frac.y);
    float f5678 = lerp_r(f56,f78,frac.y);

    retVal = lerp_r(f1234,f5678,frac.z);

    return true;
  }

  template <typename R>
  VSNRAY_FUNC
  inline hit_record<R, primitive<unsigned>> intersect(const R &ray, const UElemGrid &grid)
  {
    hit_record<R, primitive<unsigned>> result;
    float3 pos = ray.ori;
    float value = 0.f;

    bool hit = intersectGrid(grid, pos, value);
    result.hit = hit;

    if (result.hit) {
      result.t = 0.f;
      result.prim_id = grid.gridID;
      result.u = value; // misuse "u" to store value
    }

    return result;
  }


} // namespace visionaray::dco
} // namespace visionaray
