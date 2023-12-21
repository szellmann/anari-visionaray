
#include "PrimitiveSampler.h"

namespace visionaray {

PrimitiveSampler::PrimitiveSampler(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Primitive;
}

bool PrimitiveSampler::isValid() const
{
  return Sampler::isValid() && false;
}

void PrimitiveSampler::commit()
{
  Sampler::commit();

  Sampler::dispatch();
}

} // namespace visionaray
