// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/detail/color_conversion.h"
// ours
#include <common.h>

namespace visionaray {

// PixelSample //

struct PixelSample
{
  float4 color;
  float depth;
  float3 Ng;
  float3 Ns;
  float3 albedo;
  float4 motionVec;
  uint32_t primId{~0u};
  uint32_t objId{~0u};
  uint32_t instId{~0u};
};

// Helper functions ///////////////////////////////////////////////////////////

VSNRAY_FUNC
static uint32_t cvt_uint32(const float &f)
{
  return static_cast<uint32_t>(255.f * std::clamp(f, 0.f, 1.f));
}

VSNRAY_FUNC
static uint32_t cvt_uint32(const float4 &v)
{
  return (cvt_uint32(v.x) << 0) | (cvt_uint32(v.y) << 8)
      | (cvt_uint32(v.z) << 16) | (cvt_uint32(v.w) << 24);
}

VSNRAY_FUNC
static uint32_t cvt_uint32_srgb(const float4 &v)
{
  return cvt_uint32(float4(linear_to_srgb(v.xyz()), v.w));
}

} // namespace visionaray
