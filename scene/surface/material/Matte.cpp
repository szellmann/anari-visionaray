// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "scene/surface/common.h"
#include "Matte.h"

namespace visionaray {

Matte::Matte(VisionarayGlobalState *s) : Material(s) {}

void Matte::commitParameters()
{
  Material::commitParameters();

  m_color = float4(1.f, 1.f, 1.f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
  m_colorSampler = getParamObject<Sampler>("color");
  m_colorAttribute = toAttribute(getParamString("color", "none"));

  m_opacity.value = 1.f;
  getParam("opacity", ANARI_FLOAT32, &m_opacity.value);
  m_opacity.sampler = getParamObject<Sampler>("opacity");
  m_opacity.attribute = toAttribute(getParamString("opacity", "none"));

  m_alphaMode = toAlphaMode(getParamString("alphaMode", "opaque"));
  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
}

void Matte::finalize()
{
  vmat.type = dco::Material::Matte;
  vmat.asMatte.color.rgb = m_color.xyz();
  vmat.asMatte.color.attribute = m_colorAttribute;
  if (m_colorSampler && m_colorSampler->isValid()) {
    vmat.asMatte.color.samplerID = m_colorSampler->visionaraySampler().samplerID;
  } else {
    vmat.asMatte.color.samplerID = UINT_MAX;
  }

  vmat.asMatte.opacity.f = m_opacity.value;
  vmat.asMatte.opacity.attribute = m_opacity.attribute;
  if (m_opacity.sampler && m_opacity.sampler->isValid()) {
    vmat.asMatte.opacity.samplerID
        = m_opacity.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asMatte.opacity.samplerID = UINT_MAX;
  }

  vmat.asMatte.alphaMode = m_alphaMode;
  vmat.asMatte.alphaCutoff = m_alphaCutoff;

  dispatch();
}

} // namespace visionaray
