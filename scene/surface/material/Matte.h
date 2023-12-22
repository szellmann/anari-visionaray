// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Material.h"

namespace visionaray {

struct Matte : public Material
{
  Matte(VisionarayGlobalState *s);
  void commit() override;

 private:
  float4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
  dco::Attribute m_colorAttribute;
};

} // namespace visionaray
