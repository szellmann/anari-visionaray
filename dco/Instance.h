// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dco/common.h"
#include "dco/BLS.h"

namespace visionaray::dco {

// Instance //

struct Instance
{
  enum Type { Transform, MotionTransform, Unknown, };
  Type type;
  unsigned instID;
  unsigned userID;
  unsigned groupID;
  Uniform uniformAttributes[5];
#ifdef WITH_CUDA
  cuda_bvh<BLS>::bvh_ref theBVH;
#elif defined(WITH_HIP)
  hip_bvh<BLS>::bvh_ref theBVH;
#else
  bvh<BLS>::bvh_ref theBVH;
#endif
  mat4 *xfms;
  mat3 *normalXfms;
  mat3 *affineInv;
  vec3 *transInv;
  unsigned len;
  box1 time;
};

VSNRAY_FUNC
inline Instance createInstance()
{
  Instance inst;
  memset(&inst,0,sizeof(inst));
  inst.type    = Instance::Unknown;
  inst.instID  = UINT_MAX;
  inst.userID  = UINT_MAX;
  inst.groupID = UINT_MAX;
  return inst;
}

VSNRAY_FUNC
inline aabb get_bounds(const Instance &inst)
{
  if (inst.type == Instance::Transform && inst.theBVH.num_nodes()) {

    aabb bound = inst.theBVH.node(0).get_bounds();
    mat3f rot = inverse(inst.affineInv[0]);
    vec3f trans = -inst.transInv[0];
    auto verts = compute_vertices(bound);
    aabb result;
    result.invalidate();
    for (vec3 v : verts) {
      v = rot * v + trans;
      result.insert(v);
    }
    return result;
  } else if (inst.type == Instance::MotionTransform && inst.len) {
    aabb result;
    result.invalidate();
    for (unsigned i = 0; i < inst.len; ++i) {
      aabb bound = inst.theBVH.node(0).get_bounds();
      mat3f rot = inverse(inst.affineInv[i]);
      vec3f trans = -inst.transInv[i];
      auto verts = compute_vertices(bound);
      for (vec3 v : verts) {
        v = rot * v + trans;
        result.insert(v);
      }
    }
    return result;
  }

  return {};
}

inline void split_primitive(
    aabb &L, aabb &R, float plane, int axis, const Instance &inst)
{
  assert(0);
}

template <typename RayType>
VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const RayType &ray, const Instance &inst)
{
  mat3 affineInv;
  vec3 transInv;

  if (inst.type == Instance::Transform) {
    affineInv = inst.affineInv[0];
    transInv = inst.transInv[0];
  } else if (inst.type == Instance::MotionTransform) {
    float rayTime = clamp(ray.time, inst.time.min, inst.time.max);

    float time01 = rayTime - inst.time.min / (inst.time.max - inst.time.min);

    unsigned ID1 = unsigned(float(inst.len-1) * time01);
    unsigned ID2 = min(inst.len-1, ID1+1);

    float frac = time01 * (inst.len-1) - ID1;

    affineInv = lerp_r(inst.affineInv[ID1],
                       inst.affineInv[ID2],
                       frac);

    transInv = lerp_r(inst.transInv[ID1],
                      inst.transInv[ID2],
                      frac);
  }

  RayType xfmRay(ray);
  xfmRay.ori = affineInv * (xfmRay.ori + transInv);
  xfmRay.dir = affineInv * xfmRay.dir;

  auto hr = intersect(xfmRay,inst.theBVH);
  if (hr.hit) {
    hr.isect_pos = xfmRay.ori + hr.t * xfmRay.dir;
    hr.inst_id = inst.instID;
  } else {
    hr.inst_id = ~0u;
  }
  return hr;
}

} // namespace dco
