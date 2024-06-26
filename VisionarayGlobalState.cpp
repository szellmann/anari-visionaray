
#include "VisionarayGlobalState.h"
#include "frame/Frame.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), threadPool(std::thread::hardware_concurrency())
{
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamCreate(&stream));
#endif
}

VisionarayGlobalState::~VisionarayGlobalState()
{
  waitOnCurrentFrame();

#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaStreamDestroy(stream));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipStreamDestroy(stream));
#endif
}

void VisionarayGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

} // namespace visionaray
