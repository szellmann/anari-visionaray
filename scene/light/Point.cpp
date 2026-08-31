// Copyright 2023-2026 Stefan Zellmann
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
  m_radius = std::clamp(getParam<float>("radius", 0.f),
      0.f,
      std::numeric_limits<float>::max());
}

void Point::finalize()
{
  Light::finalize();
  vlight.asPoint.position = m_position;
  vlight.asPoint.color = m_color;
  vlight.asPoint.lightIntensity = m_intensity;
  vlight.asPoint.radius = m_radius;

  dispatch();
}

} // visionaray
