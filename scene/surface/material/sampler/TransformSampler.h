
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
  mat4 m_outTransform{mat4::identity()};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

} // namespace visionaray
