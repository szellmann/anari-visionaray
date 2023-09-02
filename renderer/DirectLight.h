#pragma once

#include "Renderer.h"

namespace visionaray {

struct DirectLight : public Renderer
{
  DirectLight(VisionarayGlobalState *s);
  ~DirectLight() override;
};

} // namespace visionaray
