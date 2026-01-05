// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sampler.h"

namespace visionaray {

struct TransformSampler : public Sampler
{
  TransformSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commitParameters() override;
  void finalize() override;

 private:
  dco::Attribute m_inAttribute{dco::Attribute::None};
  mat4 m_outTransform{mat4::identity()};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

} // namespace visionaray
