// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace visionaray {

struct Omnidirectional : public Camera
{
  Omnidirectional(VisionarayGlobalState *s);

  void finalize() override;
};

} // namespace visionaray
