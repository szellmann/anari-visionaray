// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Material.h"

namespace visionaray {

struct Matte : public Material
{
  Matte(VisionarayGlobalState *s);
  void commitParameters() override;
  void finalize() override;

 private:
  float4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
  dco::Attribute m_colorAttribute;

  struct {
    float value{1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_opacity;

  dco::AlphaMode m_alphaMode{dco::AlphaMode::Opaque};
  float m_alphaCutoff{0.5f};
};

} // namespace visionaray
