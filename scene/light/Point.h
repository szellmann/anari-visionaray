#pragma once

#include "Light.h"

namespace visionaray {

struct Point : public Light
{
  Point(VisionarayGlobalState *s);
  ~Point() override;

  void commit() override;

 private:
  vec3 m_position{0.f, 0.f, 0.f};
  float m_intensity{1.f};
};

} // namespace visionaray
