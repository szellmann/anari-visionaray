// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Renderer.h"

namespace visionaray {

struct Raycast : public Renderer
{
  Raycast(VisionarayGlobalState *s);
  ~Raycast() override;

  void commitParameters() override;
  void finalize() override;

 private:
  float m_volumeSamplingRate{0.5f};
};

} // visionaray
