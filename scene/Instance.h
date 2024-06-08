// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace visionaray {

struct Instance : public Object
{
  Instance(VisionarayGlobalState *s);
  ~Instance() override;

  void commit() override;

  uint32_t id() const;
  const mat4 &xfm() const;
  const mat3 &xfmInvRot() const;
  bool xfmIsIdentity() const;

  const Group *group() const;
  Group *group();

  dco::Geometry visionarayGeometry() const;
  void visionarayGeometryUpdate();

  void markCommitted() override;

  bool isValid() const override;

 private:
  uint32_t m_id{~0u};
  mat4 m_xfm;
  mat3 m_xfmInvRot;
  helium::IntrusivePtr<Group> m_group;
  dco::Geometry vgeom;

  HostDeviceArray<dco::Instance> m_instance;

  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Instance *, ANARI_INSTANCE);
