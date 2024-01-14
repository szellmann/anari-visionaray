
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

  m_ior.value = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior.value);
  m_ior.sampler = getParamObject<Sampler>("ior");
  m_ior.attribute = toAttribute(getParamString("ior", "none"));

  vmat.type = dco::Material::PhysicallyBased;

  vmat.asPhysicallyBased.baseColor.rgb = m_baseColor.value.xyz();
  vmat.asPhysicallyBased.baseColor.attribute = m_baseColor.attribute;
  if (m_baseColor.sampler) {
    vmat.asPhysicallyBased.baseColor.samplerID
        = m_baseColor.sampler->visionaraySampler().samplerID;
  }

  vmat.asPhysicallyBased.opacity.f = m_opacity.value;
  vmat.asPhysicallyBased.opacity.attribute = m_opacity.attribute;
  if (m_opacity.sampler) {
    vmat.asPhysicallyBased.opacity.samplerID
        = m_opacity.sampler->visionaraySampler().samplerID;
  }

  vmat.asPhysicallyBased.metallic.f = m_metallic.value;
  vmat.asPhysicallyBased.metallic.attribute = m_metallic.attribute;
  if (m_metallic.sampler) {
    vmat.asPhysicallyBased.metallic.samplerID
        = m_metallic.sampler->visionaraySampler().samplerID;
  }

  vmat.asPhysicallyBased.roughness.f = m_roughness.value;
  vmat.asPhysicallyBased.roughness.attribute = m_roughness.attribute;
  if (m_roughness.sampler) {
    vmat.asPhysicallyBased.roughness.samplerID
        = m_roughness.sampler->visionaraySampler().samplerID;
  }

  vmat.asPhysicallyBased.ior.f = m_ior.value;
  vmat.asPhysicallyBased.ior.attribute = m_ior.attribute;
  if (m_ior.sampler) {
    vmat.asPhysicallyBased.ior.samplerID
        = m_ior.sampler->visionaraySampler().samplerID;
  }

  dispatch();
}

} // namespace visionaray
