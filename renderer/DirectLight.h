#pragma once

#include "Renderer.h"

namespace visionaray {

struct DirectLight : public Renderer
{
  DirectLight(VisionarayGlobalState *s);
  ~DirectLight() override;

  void commitParameters() override;
  void finalize() override;
 private:
  float m_occlusionDistance{1e20f};
  int m_ambientSamples{1};
  int m_pixelSamples{1};
};

} // namespace visionaray
