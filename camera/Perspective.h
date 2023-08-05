// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace visionaray {

struct Perspective : public Camera
{
  Perspective(VisionarayGlobalState *s);

  void commit() override;

 private:
   float3 m_dir_du;
   float3 m_dir_dv;
   float3 m_dir_00;
};

} // namespace visionaray
