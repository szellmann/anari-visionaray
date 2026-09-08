// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dco/SpatialField.h"
#include "dco/DDA.h"

namespace visionaray::dco {

// Transfer functions //

struct TransferFunction1D
{
  unsigned numValues;
  box1 valueRange;
#ifdef WITH_CUDA
  cuda_texture_ref<float4, 1> sampler;
#elif defined(WITH_HIP)
  hip_texture_ref<float4, 1> sampler;
#else
  texture_ref<float4, 1> sampler;
#endif
};

VSNRAY_FUNC
inline float4 postClassify(TransferFunction1D tf, float v) {
  box1 valueRange = tf.valueRange;
  v = (v - valueRange.min) / (valueRange.max - valueRange.min);
  return tex1D(tf.sampler, v);
}

// Volume //

struct Volume
{
  enum Type { TransferFunction1D, Unknown, };
  Type type{Unknown};

  // ID in the device's global volume array
  unsigned volID{UINT_MAX};
  float unitDistance;
  bool gradientShading{false};

  SpatialField field;
  union {
    struct TransferFunction1D asTransferFunction1D;
  };

  aabb bounds;
};

VSNRAY_FUNC
inline Volume createVolume()
{
  Volume vol;
  memset(&vol,0,sizeof(vol));
  vol.type = Volume::Unknown;
  vol.volID  = UINT_MAX;
  vol.bounds.invalidate();
  return vol;
}

VSNRAY_FUNC
inline aabb get_bounds(const Volume &vol)
{
  return vol.bounds;
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const Volume &vol)
{
  assert(0);
}

struct HitRecordVolume
{
  bool hit{false};
  float t{FLT_MAX};
  float3 isect_pos;
  float3 albedo{0.f,0.f,0.f};
  float extinction{0.f};
  float Tr{1.f};
  int volID{-1}; // global to the device
  int primID{-1};
  int instID{-1};
  int localID{-1}; // local to the group
};

struct VolumePRD
{
  HitRecordVolume *hr;
  Random *rnd;
};

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(Ray ray, const Volume &vol)
{
  VolumePRD &prd = *(VolumePRD *)ray.prd;
  HitRecordVolume &hrv = *prd.hr;

  auto boxHit = intersect(ray,vol.bounds);

  hit_record<Ray, primitive<unsigned>> hr;
  hr.t = FLT_MAX;
  hr.hit = false;

  if (!boxHit.hit)
    return hr;

  if (ray.intersectionMask & Ray::VolumeBounds) {
    // we just report that we did hit the box; the user
    // is later expected to intersect the volume bounds
    // themselves to compute [t0,t1]
    hr.hit = boxHit.hit && (boxHit.tfar >= ray.tmin);
    hr.t = max(ray.tmin,boxHit.tnear);
    if (hr.t < hrv.t) {
      hrv.hit = true;
      hrv.t = hr.t;
      hrv.isect_pos = ray.ori + ray.dir * hrv.t;
      hrv.volID = vol.volID;
      hrv.primID = hr.prim_id;
    }
    return hr;
  }

  Random &rnd = *prd.rnd;

  const auto &sf = vol.field;
  dco::GridAccel grid = sf.gridAccel;

  float3 albedo;
  float Tr{1.f};
  float extinction{0.f};
  float unitDistance = vol.unitDistance;
  float invUnitDistance{1.f};

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {

    const float majorant = grid.isValid() ? grid.maxOpacities[leafID] : 1.f;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= (logf(1.f - rnd()) / (majorant * invUnitDistance));

      if (t >= t1)
        break;

      float3 P = ray.ori+ray.dir*t;
      float v = 0.f;
      int primID = 0;
      if (sampleField(sf,P,v,primID)) {
        float4 sample
            = postClassify(vol.asTransferFunction1D,v);
        float u = rnd();
        if (sample.w >= u * majorant) {
          albedo = sample.xyz();
          extinction = sample.w;
          hr.hit = true;
          Tr = 0.f;
          hr.t = t;
          hr.isect_pos = ray.ori + ray.dir * hr.t;
          hr.prim_id = primID;
          return false; // stop traversal
        }
      }
    }

    return true; // cont. traversal to the next spat. partition
  };

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  unitDistance = unitDistance * dt_scale;
  invUnitDistance = 1.f / unitDistance;

  hr.t = ray.tmax;
  if (sf.gridAccel.isValid())
    dda3(ray, grid.dims, grid.gridBounds, woodcockFunc);
  else
    woodcockFunc(-1, ray.tmin, ray.tmax);

  if (hr.hit) {
    hr.t /= dt_scale;

    if (hr.t < hrv.t) {
      hrv.hit = true;
      hrv.t = hr.t;
      hrv.isect_pos = hr.isect_pos;
      hrv.volID = vol.volID;
      hrv.localID = hr.geom_id;
      hrv.albedo = albedo;
      hrv.Tr = Tr;
      hrv.extinction = extinction;
    }
  }

  return hr;
}

} // namespace visionaray::dco
