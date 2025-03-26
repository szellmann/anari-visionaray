
#pragma once

#include "Material.h"

namespace visionaray {

struct PBM : public Material
{
  PBM(VisionarayGlobalState *s);
  void commitParameters() override;
  void finalize() override;

 private:
  struct {
    float4 value{1.f, 1.f, 1.f, 1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_baseColor;

  struct {
    float3 value{0.f, 0.f, 0.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_sheenColor;

  struct {
    float value{1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_opacity, m_metallic, m_roughness, m_clearcoat, m_clearcoatRoughness,
    m_sheenRoughness;

  float m_ior{1.5f};

  struct {
    helium::IntrusivePtr<Sampler> sampler;
    //float scale{1.f};
  } m_normal;

  dco::AlphaMode m_alphaMode{dco::AlphaMode::Opaque};
  float m_alphaCutoff{0.5f};
};

} // namespace visionaray
