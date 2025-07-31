
#include "VisionarayGlobalState.h"
#include "frame/Frame.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), threadPool(std::thread::hardware_concurrency())
{
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamCreate(&renderingStream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamCreate(&renderingStream));
#endif
}

VisionarayGlobalState::~VisionarayGlobalState()
{
  waitOnCurrentFrame();

#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamDestroy(renderingStream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamDestroy(renderingStream));
#endif
}

void VisionarayGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

} // namespace visionaray
