// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/common.h"
#include "DeviceBVH.h"
#include "Object.h"

namespace visionaray {

struct Volume : public Object
{
  Volume(VisionarayGlobalState *d);
  virtual ~Volume();
  virtual void commitParameters() override;
  uint32_t id() const;
  static Volume *createInstance(std::string_view subtype, VisionarayGlobalState *d);
  virtual aabb bounds() const = 0;
  dco::Volume visionarayVolume() const;
  dco::BLS visionarayBLS() const;

 protected:
  DeviceBVH<dco::Volume> m_BVH;

  uint32_t m_id{~0u};
  dco::Volume vvol;
  dco::BLS vBLS;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Volume *, ANARI_VOLUME);
