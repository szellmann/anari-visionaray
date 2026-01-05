// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scene/surface/geometry/Geometry.h"
#include "scene/surface/material/Material.h"

namespace visionaray {

struct Surface : public Object
{
  Surface(VisionarayGlobalState *s);
  ~Surface() override;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;
  bool isValid() const override;

  uint32_t id() const;
  Geometry *geometry();
  const Geometry *geometry() const;
  const Material *material() const;

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
