
#pragma once

#include "Material.h"

namespace visionaray {

struct PBM : public Material
{
  PBM(VisionarayGlobalState *s);
  void commit() override;

 private:
  struct {
    float4 value{1.f, 1.f, 1.f, 1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_baseColor;

  struct {
    float value{1.f};
    helium::IntrusivePtr<Sampler> sampler;
    dco::Attribute attribute;
  } m_metallic, m_roughness, m_ior;
};

} // namespace visionaray
