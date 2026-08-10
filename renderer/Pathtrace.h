// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Renderer.h"

namespace visionaray {

struct Pathtrace : public Renderer
{
  Pathtrace(VisionarayGlobalState *s);
  ~Pathtrace() override;

  void commitParameters() override;
  void finalize() override;
 private:
  int m_maxBounce{7};
  float m_occlusionDistance{1e20f};
  int m_ambientSamples{1};
  int m_pixelSamples{1};
  int m_sampleLimit{1024};
  bool m_taaEnabled{false};
  float m_taaAlpha{0.3f};
};

} // namespace visionaray
