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

  uint32_t id() const;
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

  uint32_t m_id{~0u};
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;

  dco::Surface vsurf;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Surface *, ANARI_SURFACE);
