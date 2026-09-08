// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/math/ray.h"
// ours
#include "../common.h"

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
