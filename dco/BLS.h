// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "dco/common.h"
#include "dco/Surface.h"
#include "dco/Volume.h"

namespace visionaray::dco {

// BLS primitives //

struct BLS
{
  enum Type {
    Triangle,
    Quad,
    Sphere,
    Cone,
    Cylinder,
    Curve,
    BezierCurve,
    ISOSurface,
    Volume,
    Unknown,
  };
  Type type{Unknown};
  unsigned blsID{UINT_MAX};
  // ID local to the group the BLS is in
  unsigned localID;
#ifdef WITH_CUDA
  union {
    cuda_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
    cuda_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
    cuda_bvh<basic_sphere<float>>::bvh_ref asSphere;
    cuda_bvh<dco::Cone>::bvh_ref asCone;
    cuda_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
    cuda_bvh<dco::BezierCurve>::bvh_ref asBezierCurve;
    cuda_bvh<dco::ISOSurface>::bvh_ref asISOSurface;
    cuda_bvh<dco::Volume>::bvh_ref asVolume;
  };
#elif defined(WITH_HIP)
  union {
    hip_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
    hip_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
    hip_bvh<basic_sphere<float>>::bvh_ref asSphere;
    hip_bvh<dco::Cone>::bvh_ref asCone;
    hip_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
    hip_bvh<dco::BezierCurve>::bvh_ref asBezierCurve;
    hip_bvh<dco::ISOSurface>::bvh_ref asISOSurface;
    hip_bvh<dco::Volume>::bvh_ref asVolume;
  };
#else
  union {
    bvh4<basic_triangle<3,float>>::bvh_ref asTriangle;
    bvh4<basic_triangle<3,float>>::bvh_ref asQuad;
    bvh4<basic_sphere<float>>::bvh_ref asSphere;
    bvh4<dco::Cone>::bvh_ref asCone;
    bvh4<basic_cylinder<float>>::bvh_ref asCylinder;
    bvh4<dco::BezierCurve>::bvh_ref asBezierCurve;
    bvh4<dco::ISOSurface>::bvh_ref asISOSurface;
    bvh4<dco::Volume>::bvh_ref asVolume;
  };
#endif
};

VSNRAY_FUNC
inline aabb get_bounds(const BLS &bls)
{
#ifdef WITH_HIP
  // with HIP we currenlty assume that TLSs are built on the host:
  bvh_node hip_root;
  if (bls.type == BLS::Triangle && bls.asTriangle.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asTriangle.nodes(), sizeof(hip_root), hipMemcpyDefault));
  if (bls.type == BLS::Quad && bls.asQuad.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asQuad.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Sphere && bls.asSphere.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asSphere.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Cone && bls.asCone.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asCone.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Cylinder && bls.asCylinder.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asCylinder.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::BezierCurve && bls.asBezierCurve.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asBezierCurve.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::ISOSurface && bls.asISOSurface.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asISOSurface.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Volume && bls.asVolume.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asVolume.nodes(), sizeof(hip_root), hipMemcpyDefault));
  return hip_root.get_bounds();
#else
  if (bls.type == BLS::Triangle && bls.asTriangle.num_nodes())
    return bls.asTriangle.node(0).get_bounds();
  if (bls.type == BLS::Quad && bls.asQuad.num_nodes())
    return bls.asQuad.node(0).get_bounds();
  else if (bls.type == BLS::Sphere && bls.asSphere.num_nodes())
    return bls.asSphere.node(0).get_bounds();
  else if (bls.type == BLS::Cone && bls.asCone.num_nodes())
    return bls.asCone.node(0).get_bounds();
  else if (bls.type == BLS::Cylinder && bls.asCylinder.num_nodes())
    return bls.asCylinder.node(0).get_bounds();
  else if (bls.type == BLS::BezierCurve && bls.asBezierCurve.num_nodes())
    return bls.asBezierCurve.node(0).get_bounds();
  else if (bls.type == BLS::ISOSurface && bls.asISOSurface.num_nodes())
    return bls.asISOSurface.node(0).get_bounds();
  else if (bls.type == BLS::Volume && bls.asVolume.num_nodes())
    return bls.asVolume.node(0).get_bounds();
#endif

  aabb inval;
  inval.invalidate();
  return inval;
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const BLS &bls)
{
  assert(0);
}

template <detail::traversal_type TT>
VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectBLS(const Ray &ray, const BLS &bls)
{
  hit_record<Ray, primitive<unsigned>> hr;
#if defined(WITH_CUDA) || defined(WITH_HIP)
  if (bls.type == BLS::Triangle && (ray.intersectionMask & Ray::Triangle))
    hr = intersect(ray,bls.asTriangle);
  else if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    hr = intersect(ray,bls.asQuad);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    hr = intersect(ray,bls.asSphere);
  else if (bls.type == BLS::Cone && (ray.intersectionMask & Ray::Cone))
    hr = intersect(ray,bls.asCone);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    hr = intersect(ray,bls.asCylinder);
  else if (bls.type == BLS::BezierCurve && (ray.intersectionMask & Ray::BezierCurve))
    hr = intersect(ray,bls.asBezierCurve);
  else if (bls.type == BLS::ISOSurface && (ray.intersectionMask & Ray::ISOSurface))
    hr = intersect(ray,bls.asISOSurface);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::Volume))
    hr = intersect(ray,bls.asVolume);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::VolumeBounds))
    hr = intersect(ray,bls.asVolume);
#else
  default_intersector isect;
  if (bls.type == BLS::Triangle && (ray.intersectionMask & Ray::Triangle))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asTriangle,isect);
  else if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asQuad,isect);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asSphere,isect);
  else if (bls.type == BLS::Cone && (ray.intersectionMask & Ray::Cone))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asCone,isect);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asCylinder,isect);
  else if (bls.type == BLS::BezierCurve && (ray.intersectionMask & Ray::BezierCurve))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asBezierCurve,isect);
  else if (bls.type == BLS::ISOSurface && (ray.intersectionMask & Ray::ISOSurface))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asISOSurface,isect);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::Volume))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asVolume,isect);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::VolumeBounds))
    hr = intersect_ray1_bvhN<TT>(ray,bls.asVolume,isect);
#endif

  if (hr.hit) {
    hr.geom_id = bls.localID;
  }

  return hr;
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(const Ray &ray, const BLS &bls)
{
  return intersectBLS<detail::ClosestHit>(ray, bls);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const ShadowRay &ray, const BLS &bls)
{
  return intersectBLS<detail::AnyHit>(*(Ray *)&ray, bls);
}

} // namespace visionaray::dco
