#pragma once

#include "Light.h"

namespace visionaray {

struct Directional : public Light
{
  Directional(VisionarayGlobalState *d);
  ~Directional() override;

  void commitParameters() override;
  void finalize() override;

 private:
  vec3 m_direction{0.f, 0.f, -1.f};
  float m_irradiance{1.f};
};

} // namespace visionaray
