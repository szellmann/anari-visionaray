#include "Raycast.h"

namespace visionaray {

Raycast::Raycast(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::Raycast;
}

Raycast::~Raycast()
{}

void Raycast::commitParameters()
{
  Renderer::commitParameters();
  m_volumeSamplingRate = getParam<float>("volumeSamplingRate", 0.5f);
}

void Raycast::finalize()
{
  Renderer::finalize();

  auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
  vrend.rendererState.volumeSamplingRateInv = safe_rcp(m_volumeSamplingRate);
}

} // namespace visionaray
