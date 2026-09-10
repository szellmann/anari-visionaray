// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/detail/color_conversion.h"
// ours
#include "dco/common.h"

namespace visionaray {

// ScreenSample //

struct ScreenSample
{
  int x, y;
  int frameID;
  uint2 frameSize;
  Random random;

  inline VSNRAY_FUNC bool debug() {
#if 1
    return x == frameSize.x/2 && y == frameSize.y/2;
#else
    return false;
#endif
  }
};

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


namespace dco {

// Frame //

struct Frame
{
  unsigned frameID;
  unsigned frameCounter;
  uint2 size;
  float2 invSize;
  bool stochasticRendering;

  anari::DataType colorType;
  anari::DataType depthType;
  anari::DataType normalType;
  anari::DataType albedoType;
  anari::DataType primIdType;
  anari::DataType objIdType;
  anari::DataType instIdType;

  uint32_t *pixelBuffer;
  float *depthBuffer;
  float3 *normalBuffer;
  float3 *albedoBuffer;
  float4 *motionVecBuffer;
  uint32_t *primIdBuffer;
  uint32_t *objIdBuffer;
  uint32_t *instIdBuffer;
  float4 *accumBuffer;

  struct {
    bool enabled;
    float alpha;
    float4 *currBuffer;
    float4 *prevBuffer;
    float3 *currAlbedoBuffer;
    float3 *prevAlbedoBuffer;
#ifdef WITH_CUDA
    cuda_texture_ref<float4, 2> history;
#elif defined(WITH_HIP)
    hip_texture_ref<float4, 2> history;
#else
    texture_ref<float4, 2> history;
#endif
  } taa;

  VSNRAY_FUNC
  inline PixelSample pixelSample(int x, int y) const
  {
    const auto idx = y * size.x + x;

    PixelSample s;

    if (taa.enabled) {
      if (taa.currBuffer)
        s.color = taa.currBuffer[idx];
      if (taa.currAlbedoBuffer)
        s.albedo = taa.currAlbedoBuffer[idx];
    } else {
      if (accumBuffer)
        s.color = accumBuffer[idx];
      if (albedoBuffer)
        s.albedo = albedoBuffer[idx];
    }

    if (depthBuffer)
      s.depth = depthBuffer[idx];
    if (normalBuffer)
      s.Ns = normalBuffer[idx];
    if (motionVecBuffer)
      s.motionVec = motionVecBuffer[idx];
    if (primIdBuffer)
      s.primId = primIdBuffer[idx];
    if (objIdBuffer)
      s.objId = objIdBuffer[idx];
    if (instIdBuffer)
      s.instId = instIdBuffer[idx];

    return s;
  }

  VSNRAY_FUNC
  inline PixelSample accumSample(int x, int y, int accumID, PixelSample s) const
  {
    const auto idx = y * size.x + x;

    if (taa.enabled) {
      int2 prevID = int2(float2(x,y) + motionVecBuffer[idx].xy());
      prevID = clamp(prevID, int2(0), int2(size)-int2(1));
      const auto prevIdx = prevID.y * size.x + prevID.x;
      float alpha = taa.alpha;
      if (!(fabsf(taa.prevAlbedoBuffer[prevIdx].x-taa.currAlbedoBuffer[idx].x) < 1e-2f
         && fabsf(taa.prevAlbedoBuffer[prevIdx].y-taa.currAlbedoBuffer[idx].y) < 1e-2f
         && fabsf(taa.prevAlbedoBuffer[prevIdx].z-taa.currAlbedoBuffer[idx].z) < 1e-2f)) {
        alpha = 1.f;
      }
      float prevX = x + motionVecBuffer[idx].x;
      float prevY = y + motionVecBuffer[idx].y;
      float2 texCoord((prevX+0.5f)/size.x, (prevY+0.5f)/size.y);
      float4 history = tex2D(taa.history, texCoord);
      taa.currBuffer[idx] = (1-alpha)*history + alpha*s.color;
      s.color = taa.currBuffer[idx];
    } else if (stochasticRendering) {
      float alpha = 1.f / (accumID+1);
      accumBuffer[idx] = (1-alpha)*accumBuffer[idx] + alpha*s.color;
      s.color = accumBuffer[idx];
    }

    return s;
  }

  VSNRAY_FUNC
  inline void toneMap(int x, int y, PixelSample s) const
  {
    const auto idx = y * size.x + x;

    switch (colorType) {
    case ANARI_UFIXED8_VEC4: {
      pixelBuffer[idx] = cvt_uint32(s.color);
      break;
    }
    case ANARI_UFIXED8_RGBA_SRGB: {
      pixelBuffer[idx] = cvt_uint32_srgb(s.color);
      break;
    }
    case ANARI_FLOAT32_VEC4: {
      ((float4 *)pixelBuffer)[idx] = s.color;
      break;
    }
    default:
      break;
    }
  }

  VSNRAY_FUNC
  inline void fillGBuffer(int x, int y, int accumID, PixelSample s) const
  {
    const auto idx = y * size.x + x;

    if (motionVecBuffer)
      motionVecBuffer[idx] = s.motionVec;
    if (taa.currBuffer)
      taa.currBuffer[idx] = s.color;
    if (taa.currAlbedoBuffer)
      taa.currAlbedoBuffer[idx] = s.albedo;

    // for the remaining values, only update if
    // depth is closer than the previous sample
    if (accumID > 0 && (!depthBuffer || s.depth > depthBuffer[idx]))
      return;

    if (depthBuffer)
      depthBuffer[idx] = s.depth;
    if (normalBuffer)
      normalBuffer[idx] = s.Ns;
    if (albedoBuffer)
      albedoBuffer[idx] = s.albedo;
    if (primIdBuffer)
      primIdBuffer[idx] = s.primId;
    if (objIdBuffer)
      objIdBuffer[idx] = s.objId;
    if (instIdBuffer)
      instIdBuffer[idx] = s.instId;
  }

  VSNRAY_FUNC
  inline void writeSample(int x, int y, int accumID, PixelSample s) const
  {
    fillGBuffer(x, y, accumID, s);
    toneMap(x, y, accumSample(x, y, accumID, s));
  }
};

VSNRAY_FUNC
inline Frame createFrame()
{
  Frame frame;
  memset(&frame,0,sizeof(frame));
  frame.frameID = UINT_MAX;
  frame.colorType = ANARI_UNKNOWN;
  frame.depthType = ANARI_UNKNOWN;
  frame.normalType = ANARI_UNKNOWN;
  frame.albedoType = ANARI_UNKNOWN;
  frame.primIdType = ANARI_UNKNOWN;
  frame.objIdType = ANARI_UNKNOWN;
  frame.instIdType = ANARI_UNKNOWN;
  frame.taa.alpha = 0.3f;
  return frame;
}

} // namespace dco
} // namespace visionaray
