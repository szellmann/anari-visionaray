
#pragma once

#include "Sampler.h"

namespace visionaray {

struct TransformSampler : public Sampler
{
  TransformSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commit() override;

 private:
  dco::Attribute m_inAttribute{dco::Attribute::None};
  mat4 m_transform{mat4::identity()};
};

} // namespace visionaray
