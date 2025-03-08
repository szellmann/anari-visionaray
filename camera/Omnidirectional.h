#pragma once

#include "Camera.h"

namespace visionaray {

struct Omnidirectional : public Camera
{
  Omnidirectional(VisionarayGlobalState *s);

  void finalize() override;
};

} // namespace visionaray
