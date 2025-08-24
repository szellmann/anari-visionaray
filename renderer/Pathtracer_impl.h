// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/AO.h"
#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"
#include "sampleCDF.h"
#include "VisionarayGlobalState.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  void renderFrame(const dco::Frame &frame,
                   const dco::Camera &cam,
                   uint2 size,
                   VisionarayGlobalState *state,
                   const DeviceObjectRegistry &DD,
                   const RendererState &rendererState,
                   unsigned worldID, int frameID);

  constexpr static bool stochasticRendering{true};
  constexpr static bool supportsTaa{true};
};

} // namespace visionaray
