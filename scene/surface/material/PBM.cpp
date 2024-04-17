
#include "scene/surface/common.h"
#include "PBM.h"

namespace visionaray {

PBM::PBM(VisionarayGlobalState *s) : Material(s) {}

void PBM::commit()
{
  Material::commit();

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

  m_ior = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior);

  m_normal.sampler = getParamObject<Sampler>("normal");

  m_alphaMode = toAlphaMode(getParamString("alphaMode", "opaque"));
  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);

  vmat.type = dco::Material::PhysicallyBased;

  vmat.asPhysicallyBased.baseColor.rgb = m_baseColor.value.xyz();
  vmat.asPhysicallyBased.baseColor.attribute = m_baseColor.attribute;
  if (m_baseColor.sampler) {
    vmat.asPhysicallyBased.baseColor.samplerID
        = m_baseColor.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.baseColor.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.opacity.f = m_opacity.value;
  vmat.asPhysicallyBased.opacity.attribute = m_opacity.attribute;
  if (m_opacity.sampler) {
    vmat.asPhysicallyBased.opacity.samplerID
        = m_opacity.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.opacity.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.metallic.f = m_metallic.value;
  vmat.asPhysicallyBased.metallic.attribute = m_metallic.attribute;
  if (m_metallic.sampler) {
    vmat.asPhysicallyBased.metallic.samplerID
        = m_metallic.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.metallic.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.roughness.f = m_roughness.value;
  vmat.asPhysicallyBased.roughness.attribute = m_roughness.attribute;
  if (m_roughness.sampler) {
    vmat.asPhysicallyBased.roughness.samplerID
        = m_roughness.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.roughness.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.ior = m_ior;

  if (m_normal.sampler) {
    vmat.asPhysicallyBased.normal.samplerID
        = m_normal.sampler->visionaraySampler().samplerID;
  } else {
    vmat.asPhysicallyBased.normal.samplerID = UINT_MAX;
  }

  vmat.asPhysicallyBased.alphaMode = m_alphaMode;
  vmat.asPhysicallyBased.alphaCutoff = m_alphaCutoff;

  dispatch();
}

} // namespace visionaray
