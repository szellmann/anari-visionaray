// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Quad.h"

namespace visionaray {

QuadLight::QuadLight(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::Quad;
}

QuadLight::~QuadLight()
{
}

void QuadLight::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_edge1 = getParam<vec3>("edge1", vec3(1.f, 0.f, 0.f));
  m_edge2 = getParam<vec3>("edge2", vec3(0.f, 1.f, 0.f));
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());
}

void QuadLight::finalize()
{
  Light::finalize();

  vlight.asQuad.geometry() = dco::Quad{m_position,m_edge1,m_edge2};
  vlight.asQuad.set_cl(m_color);
  vlight.asQuad.set_kl(m_intensity);

  dco::Quad temp{m_position,m_edge1,m_edge2};
  basic_triangle<3,float> t1,t2;
  temp.tessellate(t1,t2);

  dispatch();
}

} // visionaray
