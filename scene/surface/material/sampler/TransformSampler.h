
#pragma once

#include "Sampler.h"

namespace visionaray {

struct TransformSampler : public Sampler
{
  TransformSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commit() override;

 private:

};

} // namespace visionaray
