// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct Volume : public Object
{
  Volume(VisionarayGlobalState *d);
  virtual ~Volume();
  static Volume *createInstance(std::string_view subtype, VisionarayGlobalState *d);
  virtual aabb bounds() const = 0;
  // virtual void render(
  //     const VolumeRay &vray, float3 &outputColor, float &outputOpacity) = 0;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Volume *, ANARI_VOLUME);
