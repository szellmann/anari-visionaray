#include "Raycast.h"

namespace visionaray {

Raycast::Raycast(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::Raycast;
}

Raycast::~Raycast()
{}

} // namespace visionaray
