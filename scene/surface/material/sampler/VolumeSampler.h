// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sampler.h"
#include "scene/volume/Volume.h"

namespace visionaray {

struct VolumeSampler : public Sampler
{
  VolumeSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commitParameters() override;
  void finalize() override;

 private:
  helium::ChangeObserverPtr<Volume> m_volume;
  mat4 m_inTransform{mat4::identity()};
  float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  mat4 m_outTransform{mat4::identity()};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

} // namespace visionaray
