// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/World.h"

namespace visionaray {

struct PRD
{
  int x, y;
};

struct PixelSample
{
  float4 color;
  float depth;
};

struct VisionarayRenderer
{
  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd) {
    PixelSample result;
    result.color = float4(ray.dir,1.f);
    result.depth = 1.f;
    return result;
  }

  float4 m_bgColor{float3(0.f), 1.f};
  float m_ambientRadiance{1.f};
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer visionarayRenderer() const { return vrend; }

 private:
  VisionarayRenderer vrend;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
