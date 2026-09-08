// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visionaray/bvh.h"
// ours
#include "dco/common.h"
#include "dco/Instance.h"

namespace visionaray::dco {

// TLS //

#ifdef WITH_CUDA
typedef cuda_bvh<Instance>::bvh_ref TLS;
#elif defined(WITH_HIP)
typedef hip_bvh<Instance>::bvh_ref TLS;
#else
typedef bvh<Instance>::bvh_ref TLS;
#endif

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    Ray ray, const TLS &tls, bool shadow)
{
  ray.intersectionMask
      = Ray::Triangle | Ray::Quad | Ray::Sphere | Ray::Cone | Ray::Cylinder |
        Ray::Curve | Ray::BezierCurve | Ray::ISOSurface;
  if (shadow) {
    ShadowRay shadowRay = *(ShadowRay *)&ray;
    return intersect(shadowRay, tls);
  } else {
    return intersect(ray, tls);
  }
}

VSNRAY_FUNC
inline HitRecordVolume intersectVolumeBounds(Ray ray, const TLS &tls)
{
  HitRecordVolume result;

  VolumePRD prd;
  prd.hr = &result;
  ray.prd = &prd;

  ray.intersectionMask = Ray::VolumeBounds;

  auto hr = intersect(ray, tls);

  result.primID = hr.prim_id;
  result.instID = hr.inst_id;
  result.localID = hr.geom_id;

  return result;
}

VSNRAY_FUNC
inline HitRecordVolume intersectVolumes(Ray ray, const TLS &tls)
{
  HitRecordVolume result;

  VolumePRD prd;
  prd.hr = &result;
  prd.rnd = (Random *)ray.prd;
  ray.prd = &prd;

  ray.intersectionMask = Ray::Volume;
  auto hr = intersect(ray, tls);

  result.primID = hr.prim_id;
  result.instID = hr.inst_id;
  result.isect_pos = hr.isect_pos;

  return result;
}

} // namespace visionaray::dco
