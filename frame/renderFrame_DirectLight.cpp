
#include "renderFrame.h"

namespace visionaray {

void renderFrame_DirectLight(const dco::Frame &frame,
                             const dco::Camera &cam,
                             const VisionarayRenderer &rend,
                             uint2 size,
                             VisionarayGlobalState *state,
                             const VisionarayGlobalState::DeviceObjectRegistry &DD,
                             unsigned worldID, int frameID, int spp)
{
  renderFrame(
      frame, cam, rend.asDirectLight.renderer, size, state, DD, worldID, frameID, spp);
}

} // namespace visionaray
