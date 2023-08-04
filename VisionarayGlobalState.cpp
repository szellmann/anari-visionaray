
#include "VisionarayGlobalState.h"
#include "frame/Frame.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
{}

void VisionarayGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

} // namespace visionaray
