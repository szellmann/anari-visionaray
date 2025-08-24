// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Point.h"

namespace visionaray {

Point::Point(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::Point;
}

Point::~Point()
{
}

void Point::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, -1.f));
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());
}

void Point::finalize()
{
  Light::finalize();
  vlight.asPoint.set_position(m_position);
  vlight.asPoint.set_cl(m_color);
  vlight.asPoint.set_kl(m_intensity);
  vlight.asPoint.set_constant_attenuation(1.f);
  vlight.asPoint.set_linear_attenuation(0.f);
  vlight.asPoint.set_quadratic_attenuation(0.f);

  dispatch();
}

} // visionaray
