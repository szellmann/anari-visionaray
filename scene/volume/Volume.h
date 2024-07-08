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
  virtual void commit() override;
  uint32_t id() const;
  static Volume *createInstance(std::string_view subtype, VisionarayGlobalState *d);
  virtual aabb bounds() const = 0;
  dco::Geometry visionarayGeometry() const;

  helium::TimeStamp lastUpdateRequest{0};
  helium::TimeStamp lastUpdate{0};
 protected:
  uint32_t m_id{~0u};
  dco::Geometry vgeom;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Volume *, ANARI_VOLUME);
