// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dco/SpatialField.h"
#include "dco/DDA.h"

namespace visionaray::dco {

// Gaussian primitive //

struct Gaussian : public primitive<unsigned>
{
  float3 pos;
  quatf rot;
  float3 scale;
  float alpha;
};

// Gaussian splat volume //

struct GaussianSplat
{
#ifdef WITH_CUDA
  cuda_cwbvh<Gaussian>::bvh_ref splatBVH;
#elif defined(WITH_HIP)
  hip_bvh<Gaussian>::bvh_ref splatBVH;
#else
  bvh4<Gaussian>::bvh_ref splatBVH;
#endif
  Array primitives;
  Array colors;
};


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
  enum Type { GaussianSplat, TransferFunction1D, Unknown, };
  Type type{Unknown};

  // ID in the device's global volume array
  unsigned volID{UINT_MAX};
  float unitDistance;
  bool gradientShading{false};

  SpatialField field;
  union {
    struct GaussianSplat asGaussianSplat;
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

// Gaussian splat intersector //

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(Ray r, const Gaussian &g)
{
  auto xfm = [g](Ray r) {
    mat3 S = mat3::scaling(g.scale);
    mat4 R4 = rotation(normalize(g.rot));
    mat3 R(R4.col0.xyz(),R4.col1.xyz(),R4.col2.xyz());

    mat3 SI = inverse(S);
    mat3 RI = transpose(R);

    mat3 M_inv = SI * RI;

    r.ori = M_inv * (r.ori- g.pos);
    r.dir = M_inv * r.dir;

    return r;
  };

  // transform ray into reference system where Gaussian is a unit sphere
  r = xfm(r);

  const float dt_scale = length(r.dir);
  r.dir = normalize(r.dir);

  r.tmin = r.tmin * dt_scale;
  r.tmax = r.tmax * dt_scale;

  float radius = 3.f;

  float3 oc = r.ori;

  auto A = dot(r.dir, r.dir);
  auto B = dot(r.dir, oc);
  // numerically more stable version for large distance between ray origin and sphere center
  auto t_local = -B / A;
  auto p_local = oc + t_local * r.dir;
  auto C_local = radius * radius - dot(p_local, p_local);

  bool valid = C_local >= 0.f;

  auto root_disc = sqrtf(fmaxf(0.f, C_local / A));

  auto t1 = t_local - root_disc;
  auto t2 = t_local + root_disc;

  hit_record<Ray, primitive<unsigned>> result;

  if (!valid || t2 < r.tmin || t1 > r.tmax) {
    result.hit = false;
    return result;
  }

  float tnear = clamp(t1,r.tmin,r.tmax);
  float tfar  = clamp(t2,r.tmin,r.tmax);

  VolumePRD &prd = *(VolumePRD *)r.prd;
  Random &rnd = *prd.rnd;

  auto woodcock = [&](float t0, float t1) {
    const float majorant = g.alpha;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= (logf(1.f - rnd()) / (majorant));

      if (t > t1)
        break;

      float3 P = r.ori+r.dir*t;
      float r2 = dot(P,P);
      float w = g.alpha*expf(-0.5f*r2);
      float u = rnd();
      if (w >= u * majorant) {
        return t;
      }
    }
    return FLT_MAX;
  };

  float t = woodcock(tnear, tfar);

  result.prim_id = g.prim_id;
  result.geom_id = g.geom_id;
  result.t = t / dt_scale;
  result.isect_pos = r.ori + r.dir*result.t;
  result.hit = t < FLT_MAX;
  return result;
}

VSNRAY_FUNC inline aabb get_bounds(const Gaussian &g)
{
  mat3 S = mat3::scaling(g.scale);
  mat4 R4 = rotation(normalize(g.rot));
  mat3 R(R4.col0.xyz(), R4.col1.xyz(), R4.col2.xyz());

  mat3 M = R * S;

  constexpr float N = 3.0f;

  float3 radius;
  radius.x = N * sqrtf(M(0,0)*M(0,0) + M(0,1)*M(0,1) + M(0,2)*M(0,2));
  radius.y = N * sqrtf(M(1,0)*M(1,0) + M(1,1)*M(1,1) + M(1,2)*M(1,2));
  radius.z = N * sqrtf(M(2,0)*M(2,0) + M(2,1)*M(2,1) + M(2,2)*M(2,2));

  aabb result;
  result.min = g.pos - radius;
  result.max = g.pos + radius;
  
  return result;
}

VSNRAY_FUNC inline void split_primitive(
    aabb& L, aabb& R, float plane, int axis, const Gaussian &g)
{
  VSNRAY_UNUSED(L);
  VSNRAY_UNUSED(R);
  VSNRAY_UNUSED(plane);
  VSNRAY_UNUSED(axis);
  VSNRAY_UNUSED(g);

  // TODO: implement this to support SBVHs
}

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

  // Intersect as Gaussian Splat //

  if (vol.type == Volume::GaussianSplat) {
#ifdef WITH_CUDA
    default_intersector isect;
    auto hr = intersect_ray1_cwbvh<detail::ClosestHit>(ray,
                                                       vol.asGaussianSplat.splatBVH,
                                                       isect);
#elif defined(WITH_HIP)
    auto hr = intersect(ray, vol.asGaussianSplat.splatBVH);
#else
    default_intersector isect;
    auto hr = intersect_ray1_bvhN<detail::ClosestHit>(ray,
                                                      vol.asGaussianSplat.splatBVH,
                                                      isect);
#endif

    if (hr.hit && hr.t < hrv.t) {
      hrv.hit = true;
      hrv.t = hr.t;
      hrv.isect_pos = hr.isect_pos;
      hrv.volID = vol.volID;
      hrv.localID = 0;//hr.geom_id;
      hrv.albedo = ((float3 *)vol.asGaussianSplat.colors.data)[hr.prim_id];
    }
    return hr;
  }

  // Intersect as TF1D //

  Random &rnd = *prd.rnd;

  const auto &sf = vol.field;
  dco::GridAccel grid = sf.gridAccel;

  float3 albedo;
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
          hr.hit = true;
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
    }
  }

  return hr;
}

} // namespace visionaray::dco
