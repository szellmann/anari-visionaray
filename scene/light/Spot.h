// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"

namespace visionaray {

struct Spot : public Light
{
  Spot(VisionarayGlobalState *s);
  ~Spot() override;

  void commitParameters() override;
  void finalize() override;

 private:
  vec3 m_position{0.f, 0.f, 0.f};
  vec3 m_direction{0.f, 0.f, -1.f};
  float m_openingAngle{M_PI};
  float m_falloffAngle{0.1f};
  float m_intensity{1.f};
};

} // namespace visionaray
