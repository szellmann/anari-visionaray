// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/texture/texture.h"
// ours
#include "dco/common.h"

namespace visionaray::dco {

// Sampler //

struct Sampler
{
  enum Type { Image1D, Image2D, Image3D, Transform, Primitive, Volume, Unknown, };
  Type type;
  unsigned samplerID;
  Attribute inAttribute;
  mat4 inTransform;
  float4 inOffset;
  mat4 outTransform;
  float4 outOffset;
  union {
#ifdef WITH_CUDA
    cuda_texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    cuda_texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    cuda_texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#elif defined(WITH_HIP)
    hip_texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    hip_texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    hip_texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#else
    texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#endif
    struct {
      TypeInfo typeInfo;
      size_t len; // in elements
      const uint8_t *data;
      uint32_t offset;
    } asPrimitive;
    struct {
      unsigned volID;
    } asVolume;
  };

  VSNRAY_FUNC
  bool isValid() const
  {
    return samplerID < UINT_MAX &&
        inAttribute != Attribute::None &&
        (type == Image1D && asImage1D) ||
        (type == Image2D && asImage2D) ||
        (type == Image3D && asImage3D) ||
        (type == Primitive && asPrimitive.data) ||
        (type == Volume && validHandle(asVolume.volID)) ||
        (type == Transform);
  }
};

VSNRAY_FUNC
inline Sampler createSampler()
{
  Sampler samp;
  memset(&samp,0,sizeof(samp));
  samp.type = Sampler::Unknown;
  samp.samplerID = UINT_MAX;
  samp.inAttribute = Attribute::_0;
  samp.inTransform = mat4::identity();
  samp.inOffset = float4(0.f);
  samp.outTransform = mat4::identity();
  samp.outOffset = float4(0.f);
  return samp;
}

} // namespace visionaray::dco
