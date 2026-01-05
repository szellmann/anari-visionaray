// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"

namespace visionaray {

struct QuadLight : public Light
{
  QuadLight(VisionarayGlobalState *s);
  ~QuadLight() override;

  void commitParameters() override;
  void finalize() override;

 private:
  vec3 m_position{0.f, 0.f, 0.f};
  vec3 m_edge1{1.f, 0.f, 0.f};
  vec3 m_edge2{0.f, 1.f, 0.f};
  float m_intensity{1.f};
};

} // namespace visionaray
