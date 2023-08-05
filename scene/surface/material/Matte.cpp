// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Matte.h"

namespace visionaray {

Matte::Matte(VisionarayGlobalState *s) : Material(s) {}

void Matte::commit()
{
  Material::commit();

  m_color = float4(1.f, 1.f, 1.f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
  m_colorSampler = getParamObject<Sampler>("color");
}

} // namespace visionaray
