
#pragma once

#include "Sampler.h"

namespace visionaray {

struct PrimitiveSampler : public Sampler
{
  PrimitiveSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commit() override;

 private:

};

} // namespace visionaray
