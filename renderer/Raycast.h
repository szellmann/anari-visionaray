#pragma once

#include "Renderer.h"

namespace visionaray {

struct Raycast : public Renderer
{
  Raycast(VisionarayGlobalState *s);
  ~Raycast() override;
};

} // visionaray
