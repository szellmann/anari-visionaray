// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/common.h"
#include "Object.h"

namespace visionaray {

struct Volume : public Object
{
  Volume(VisionarayGlobalState *d);
  virtual ~Volume();
  static Volume *createInstance(std::string_view subtype, VisionarayGlobalState *d);
  virtual aabb bounds() const = 0;
  dco::Geometry visionarayGeometry() const;

 protected:

  dco::Geometry vgeom;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Volume *, ANARI_VOLUME);
