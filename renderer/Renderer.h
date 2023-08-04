// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/World.h"

namespace visionaray {

struct PixelSample
{
  float4 color;
  float depth;
};

enum class RenderMode
{
  DEFAULT,
  PRIM_ID,
  GEOM_ID,
  INST_ID,
  NG,
  NG_ABS,
  NS,
  NS_ABS,
  RAY_UVW,
  HIT_SURFACE,
  HIT_VOLUME,
  BACKFACE,
  HAS_MATERIAL,
  GEOMETRY_ATTRIBUTE_0,
  GEOMETRY_ATTRIBUTE_1,
  GEOMETRY_ATTRIBUTE_2,
  GEOMETRY_ATTRIBUTE_3,
  GEOMETRY_ATTRIBUTE_COLOR,
  OPACITY_HEATMAP
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

//  PixelSample renderSample(Ray ray, const World &w) const;
//
  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

// private:
//  float3 shadeRay(const Ray &ray, const VolumeRay &vray, const World &w) const;
//
//  float4 m_bgColor{float3(0.f), 1.f};
//  float m_ambientRadiance{1.f};
//  RenderMode m_mode{RenderMode::DEFAULT};
//
//  helium::IntrusivePtr<Array1D> m_heatmap;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
