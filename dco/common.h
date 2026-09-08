// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/math/ray.h"
#include "visionaray/bvh.h"
// ours
#include "../common.h"

#if defined(WITH_CUDA) && !defined(__CUDACC__)
#include <visionaray/cuda/device_vector.h>
namespace visionaray {
// visionaray only defines these when compiling with nvcc:
template <typename P>
using cuda_bvh          = bvh_t<cuda::device_vector<P>, cuda::device_vector<bvh_node>>;
} // namespace visionaray
#endif

#if defined(WITH_HIP) && !defined(__HIPCC__)
#include <visionaray/hip/device_vector.h>
namespace visionaray {
// visionaray only defines these when compiling with hipcc:
template <typename P>
using hip_bvh           = bvh_t<hip::device_vector<P>, hip::device_vector<bvh_node>>;
} // namespace visionaray
#endif

namespace visionaray
{

// Ray //

struct Ray : basic_ray<float>
{
  enum IntersectionMask {
    All = 0xffffffff,
    Triangle = 0x1,
    Quad = 0x2,
    Sphere = 0x4,
    Cone = 0x8,
    Cylinder = 0x10,
    Curve = 0x20,
    BezierCurve = 0x40,
    ISOSurface = 0x80,
    Volume = 0x100,
    VolumeBounds = 0x200,
  };
  unsigned intersectionMask = (unsigned)All;
  float time{0.f};
  void *prd{nullptr};

#if 1
  bool dbg{false};
  VSNRAY_FUNC inline bool debug() const {
    return dbg;
  }
#endif
};

struct ShadowRay : Ray {};


// Handle //

namespace dco  {
typedef uint32_t Handle;
VSNRAY_FUNC
inline bool validHandle(Handle hnd)
{ return hnd < UINT_MAX; }
} // namespace dco
typedef dco::Handle DeviceObjectHandle;

namespace dco {

// Uniform //

struct Uniform
{
  float4 value;
  bool isSet;
};

// Array //

struct Array
{
  const void *data;
  size_t len;
  TypeInfo typeInfo;
};

} // namespace dco

} // namespace visionaray
