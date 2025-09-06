// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "scene/surface/common.h"
#include "PBM.h"

namespace visionaray {

PBM::PBM(VisionarayGlobalState *s) : Material(s) {}

void PBM::commitParameters()
{
  Material::commitParameters();

  m_baseColor.value = float4(1.f, 1.f, 1.f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor.value);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor.value);
  m_baseColor.sampler = getParamObject<Sampler>("baseColor");
  m_baseColor.attribute = toAttribute(getParamString("baseColor", "none"));

  m_opacity.value = 1.f;
  getParam("opacity", ANARI_FLOAT32, &m_opacity.value);
  m_opacity.sampler = getParamObject<Sampler>("opacity");
  m_opacity.attribute = toAttribute(getParamString("opacity", "none"));

  m_metallic.value = 1.f;
  getParam("metallic", ANARI_FLOAT32, &m_metallic.value);
  m_metallic.sampler = getParamObject<Sampler>("metallic");
  m_metallic.attribute = toAttribute(getParamString("metallic", "none"));

  m_roughness.value = 1.f;
  getParam("roughness", ANARI_FLOAT32, &m_roughness.value);
  m_roughness.sampler = getParamObject<Sampler>("roughness");
  m_roughness.attribute = toAttribute(getParamString("roughness", "none"));

  m_anisotropyStrength.value = 0.f;
  getParam("anisotropyStrength", ANARI_FLOAT32, &m_anisotropyStrength.value);
  m_anisotropyStrength.sampler = getParamObject<Sampler>("anisotropyStrength");
  m_anisotropyStrength.attribute
      = toAttribute(getParamString("anisotropyStrength", "none"));

  m_anisotropyDirection.value = float2(1.f, 0.f);
  getParam("anisotropyDirection", ANARI_FLOAT32_VEC2, &m_anisotropyDirection.value);
  m_anisotropyDirection.sampler = getParamObject<Sampler>("anisotropyDirection");
  m_anisotropyDirection.attribute
      = toAttribute(getParamString("anisotropyDirection", "none"));

  m_anisotropyRotation.value = 0.f;
  getParam("anisotropyRotation", ANARI_FLOAT32, &m_anisotropyRotation.value);
  m_anisotropyRotation.sampler = getParamObject<Sampler>("anisotropyRotation");
  m_anisotropyRotation.attribute
      = toAttribute(getParamString("anisotropyRotation", "none"));

  m_transmission.value = 0.f;
  getParam("transmission", ANARI_FLOAT32, &m_transmission.value);
  m_transmission.sampler = getParamObject<Sampler>("transmission");
  m_transmission.attribute = toAttribute(getParamString("transmission", "none"));

  m_normal.sampler = getParamObject<Sampler>("normal");

  m_clearcoat.value = 0.f;
  getParam("clearcoat", ANARI_FLOAT32, &m_clearcoat.value);
  m_clearcoat.sampler = getParamObject<Sampler>("clearcoat");
  m_clearcoat.attribute = toAttribute(getParamString("clearcoat", "none"));

  m_clearcoatRoughness.value = 0.f;
  getParam("clearcoatRoughness", ANARI_FLOAT32, &m_clearcoatRoughness.value);
  m_clearcoatRoughness.sampler = getParamObject<Sampler>("clearcoatRoughness");
  m_clearcoatRoughness.attribute
      = toAttribute(getParamString("clearcoatRoughness", "none"));

  m_ior = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior);

  m_sheenColor.value = float3(0.f, 0.f, 0.f);
  getParam("sheenColor", ANARI_FLOAT32_VEC3, &m_sheenColor.value);
  m_sheenColor.sampler = getParamObject<Sampler>("sheenColor");
  m_sheenColor.attribute = toAttribute(getParamString("sheenColor", "none"));

  m_sheenRoughness.value = 0.f;
  getParam("sheenRoughness", ANARI_FLOAT32, &m_sheenRoughness.value);
  m_sheenRoughness.sampler = getParamObject<Sampler>("sheenRoughness");
  m_sheenRoughness.attribute = toAttribute(getParamString("sheenRoughness", "none"));

  m_alphaMode = toAlphaMode(getParamString("alphaMode", "opaque"));
  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
}

void PBM::finalize()
{
  vmat.type = dco::Material::PhysicallyBased;

  vmat.asPhysicallyBased.baseColor.rgb = m_baseColor.value.xyz();
  vmat.asPhysicallyBased.baseColor.attribute = m_baseColor.attribute;
  if (m_baseColor.sampler && m_baseColor.sampler->isValid()) {
    vmat.asPhysicallyBased.baseColor.samplerID
        = m_baseColor.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.baseColor.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.opacity.f = m_opacity.value;
  vmat.asPhysicallyBased.opacity.attribute = m_opacity.attribute;
  if (m_opacity.sampler && m_opacity.sampler->isValid()) {
    vmat.asPhysicallyBased.opacity.samplerID
        = m_opacity.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.opacity.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.metallic.f = m_metallic.value;
  vmat.asPhysicallyBased.metallic.attribute = m_metallic.attribute;
  if (m_metallic.sampler && m_metallic.sampler->isValid()) {
    vmat.asPhysicallyBased.metallic.samplerID
        = m_metallic.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.metallic.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.roughness.f = m_roughness.value;
  vmat.asPhysicallyBased.roughness.attribute = m_roughness.attribute;
  if (m_roughness.sampler && m_roughness.sampler->isValid()) {
    vmat.asPhysicallyBased.roughness.samplerID
        = m_roughness.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.roughness.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.anisotropyStrength.f = m_anisotropyStrength.value;
  vmat.asPhysicallyBased.anisotropyStrength.attribute = m_anisotropyStrength.attribute;
  if (m_anisotropyStrength.sampler && m_anisotropyStrength.sampler->isValid()) {
    vmat.asPhysicallyBased.anisotropyStrength.samplerID
        = m_anisotropyStrength.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.anisotropyStrength.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.anisotropyDirection.uv = m_anisotropyDirection.value;
  vmat.asPhysicallyBased.anisotropyDirection.attribute = m_anisotropyDirection.attribute;
  if (m_anisotropyDirection.sampler && m_anisotropyDirection.sampler->isValid()) {
    vmat.asPhysicallyBased.anisotropyDirection.samplerID
        = m_anisotropyDirection.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.anisotropyDirection.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.anisotropyRotation.f = m_anisotropyRotation.value;
  vmat.asPhysicallyBased.anisotropyRotation.attribute = m_anisotropyRotation.attribute;
  if (m_anisotropyRotation.sampler && m_anisotropyRotation.sampler->isValid()) {
    vmat.asPhysicallyBased.anisotropyRotation.samplerID
        = m_anisotropyRotation.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.anisotropyRotation.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.transmission.f = m_transmission.value;
  vmat.asPhysicallyBased.transmission.attribute = m_transmission.attribute;
  if (m_transmission.sampler && m_transmission.sampler->isValid()) {
    vmat.asPhysicallyBased.transmission.samplerID
        = m_transmission.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.transmission.samplerID = UINT_MAX;
  }

  if (m_normal.sampler && m_normal.sampler->isValid()) {
    vmat.asPhysicallyBased.normal.samplerID
        = m_normal.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.normal.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.clearcoat.f = m_clearcoat.value;
  vmat.asPhysicallyBased.clearcoat.attribute = m_clearcoat.attribute;
  if (m_clearcoat.sampler && m_clearcoat.sampler->isValid()) {
    vmat.asPhysicallyBased.clearcoat.samplerID
        = m_clearcoat.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.clearcoat.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.clearcoatRoughness.f = m_clearcoatRoughness.value;
  vmat.asPhysicallyBased.clearcoatRoughness.attribute = m_clearcoatRoughness.attribute;
  if (m_clearcoatRoughness.sampler && m_clearcoatRoughness.sampler->isValid()) {
    vmat.asPhysicallyBased.clearcoatRoughness.samplerID
        = m_clearcoatRoughness.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.clearcoatRoughness.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.ior = m_ior;

  vmat.asPhysicallyBased.sheenColor.rgb = m_sheenColor.value;
  vmat.asPhysicallyBased.sheenColor.attribute = m_sheenColor.attribute;
  if (m_sheenColor.sampler && m_sheenColor.sampler->isValid()) {
    vmat.asPhysicallyBased.sheenColor.samplerID
        = m_sheenColor.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.sheenColor.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.sheenRoughness.f = m_sheenRoughness.value;
  vmat.asPhysicallyBased.sheenRoughness.attribute = m_sheenRoughness.attribute;
  if (m_sheenRoughness.sampler && m_sheenRoughness.sampler->isValid()) {
    vmat.asPhysicallyBased.sheenRoughness.samplerID
        = m_sheenRoughness.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.sheenRoughness.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.alphaMode = m_alphaMode;
  vmat.asPhysicallyBased.alphaCutoff = m_alphaCutoff;

  dispatch();
}

} // namespace visionaray
