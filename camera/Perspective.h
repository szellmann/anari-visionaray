// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace visionaray {

struct Perspective : public Camera
{
  Perspective(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:
   float m_fovy{0.f};
   float m_aspect{1.f};
   float m_apertureRadius{0.f};
   float m_focusDistance{1.f};
   float3 m_dir_du;
   float3 m_dir_dv;
   float3 m_dir_00;
};

} // namespace visionaray
