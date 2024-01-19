
#include "renderFrame.h"

namespace visionaray {

void renderFrame_Raycast(const dco::Frame &frame,
                         const dco::Camera &cam,
                         const VisionarayRenderer &rend,
                         uint2 size,
                         VisionarayGlobalState *state,
                         const VisionarayGlobalState::DeviceObjectRegistry &DD,
                         unsigned worldID, int frameID, int spp)
{
  renderFrame(
      frame, cam, rend.asRaycast.renderer, size, state, DD, worldID, frameID, spp);
}

} // namespace visionaray
