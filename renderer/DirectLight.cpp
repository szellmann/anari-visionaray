#include "DirectLight.h"

namespace visionaray {

DirectLight::DirectLight(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::DirectLight;
}

DirectLight::~DirectLight()
{}

} // namespace visionaray
