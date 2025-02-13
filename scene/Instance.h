// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace visionaray {

struct Instance : public Object
{
  Instance(VisionarayGlobalState *s);
  ~Instance() override;

  static Instance *createInstance(
      std::string_view subtype, VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;
  bool isValid() const override;


  uint32_t id() const;

  const Group *group() const;
  Group *group();

  dco::Instance visionarayInstance() const;
  virtual void visionarayInstanceUpdate();

 protected:
  mat4 m_xfm;
  helium::ChangeObserverPtr<Array1D> m_xfmArray;

  uint32_t m_id{~0u};
  helium::ChangeObserverPtr<Array1D> m_idArray;

  helium::IntrusivePtr<Group> m_group;

  dco::Instance vinstance;

  HostDeviceArray<mat4> m_xfms;
  HostDeviceArray<mat3> m_normalXfms;
  HostDeviceArray<mat3> m_affineInv;
  HostDeviceArray<vec3> m_transInv;

  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Instance *, ANARI_INSTANCE);
