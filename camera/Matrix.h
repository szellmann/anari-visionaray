// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace visionaray {

struct Matrix : public Camera
{
  Matrix(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:
  mat4 m_proj;
  mat4 m_view;
};

} // namespace visionaray
