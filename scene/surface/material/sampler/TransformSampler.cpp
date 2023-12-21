
#include "TransformSampler.h"

namespace visionaray {

TransformSampler::TransformSampler(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Transform;
}

bool TransformSampler::isValid() const
{
  return Sampler::isValid() && false;
}

void TransformSampler::commit()
{
  Sampler::commit();

  Sampler::dispatch();
}

} // namespace visionaray
