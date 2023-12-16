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
  mat4 m_xfm;
  mat3 m_xfmInvRot;
  helium::IntrusivePtr<Group> m_group;
  dco::Geometry vgeom;

  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Instance *, ANARI_INSTANCE);
