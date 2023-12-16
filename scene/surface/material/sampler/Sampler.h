// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct Geometry;

struct Sampler : public Object
{
  Sampler(VisionarayGlobalState *d);
  virtual ~Sampler();

  static Sampler *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  dco::Sampler visionaraySampler() const;

 protected:
  void dispatch();

  dco::Sampler vsampler;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Sampler *, ANARI_SAMPLER);
