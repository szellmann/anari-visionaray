// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
// space skipping
#include "GridAccel.h"

namespace visionaray {

struct SpatialField : public Object
{
  SpatialField(VisionarayGlobalState *d);
  virtual ~SpatialField();
  static SpatialField *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  dco::SpatialField visionaraySpatialField() const;

  GridAccel &gridAccel();

  virtual aabb bounds() const = 0;

  virtual void buildGrid();

 protected:
  dco::SpatialField vfield;

  GridAccel m_gridAccel;

  void setCellSize(float cellSize);
  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(
    visionaray::SpatialField *, ANARI_SPATIAL_FIELD);
