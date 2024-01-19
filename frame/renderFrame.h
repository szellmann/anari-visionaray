#pragma once

#include "DeviceCopyableObjects.h"

namespace visionaray {

void renderFrame(const dco::Frame &frame,
                 const dco::Camera &cam,
                 uint2 size,
                 VisionarayRenderer &rend,
                 VisionarayGlobalState *state,
                 const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
                 unsigned worldID,
                 int frameID);

} // namespace visionaray
