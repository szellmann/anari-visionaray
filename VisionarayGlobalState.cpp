
#include "VisionarayGlobalState.h"
#include "frame/Frame.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), threadPool(std::thread::hardware_concurrency())
{}

void VisionarayGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

} // namespace visionaray
