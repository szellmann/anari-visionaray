// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Directional.h"

namespace visionaray {

Directional::Directional(VisionarayGlobalState *d) : Light(d)
{
  vlight.type = dco::Light::Directional;
}

Directional::~Directional()
{
}

void Directional::commitParameters()
{
  Light::commitParameters();
  m_direction =
      normalize(getParam<vec3>("direction", vec3(0.f, 0.f, -1.f)));
  m_irradiance = std::clamp(getParam<float>("irradiance", 1.f),
      0.f,
      std::numeric_limits<float>::max());
}

void Directional::finalize()
{
  Light::finalize();

  vlight.asDirectional.set_direction(-m_direction);
  vlight.asDirectional.set_cl(m_color);
  vlight.asDirectional.set_kl(m_irradiance); // TODO!
  vlight.asDirectional.set_angular_diameter(3.f); // TODO!

  dispatch();
}

} // namespace visionaray
