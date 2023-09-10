// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct SpatialField : public Object
{
  SpatialField(VisionarayGlobalState *d);
  virtual ~SpatialField();
  static SpatialField *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  dco::SpatialField visionaraySpatialField() const;

  virtual aabb bounds() const = 0;

  float stepSize() const;

 protected:
  dco::SpatialField vfield;

  void setStepSize(float size);
  void dispatch();
  void detach();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(
    visionaray::SpatialField *, ANARI_SPATIAL_FIELD);
