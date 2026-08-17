// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"
#include "sampleCDF.h"
#include "SyncContext.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  void renderFrame(DevicePointer<DeviceObjectRegistry> onDevicePtr,
                   DevicePointer<RendererState> rendererStatePtr,
                   DevicePointer<dco::Frame> framePtr,
                   DevicePointer<dco::Camera> camPtr,
                   uint2 size,
                   SyncContext::SP syncContext,
                   unsigned worldID, int frameID);

  constexpr static bool stochasticRendering{true};
  constexpr static bool supportsTaa{true};
};

} // namespace visionaray
