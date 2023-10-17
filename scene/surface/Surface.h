// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scene/surface/geometry/Geometry.h"
#include "scene/surface/material/Material.h"

namespace visionaray {

struct Surface : public Object
{
  Surface(VisionarayGlobalState *s);
  ~Surface() override;

  void commit() override;

  const Geometry *geometry() const;
  const Material *material() const;

  // float4 getSurfaceColor(const Ray &ray) const;
  // float getSurfaceOpacity(const Ray &ray) const;

  // float adjustedAlpha(float a) const;

  void markCommitted() override;
  bool isValid() const override;

  dco::Surface visionaraySurface() const { return vsurf; }

 private:
  void dispatch();
  void detach();

  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;

  dco::Surface vsurf;
};

// Inlined definitions ////////////////////////////////////////////////////////

// inline float Surface::adjustedAlpha(float a) const
// {
//   if (!material())
//     return 0.f;
// 
//   return adjustOpacityFromMode(
//       a, material()->alphaCutoff(), material()->alphaMode());
// }

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Surface *, ANARI_SURFACE);
