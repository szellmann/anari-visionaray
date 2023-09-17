// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct Light : public Object
{
  Light(VisionarayGlobalState *d);
  virtual ~Light();

  virtual void commit() override;

  static Light *createInstance(std::string_view subtype, VisionarayGlobalState *d);

  dco::Light visionarayLight() const { return vlight; }

 protected:
  dco::Light vlight;
  vec3 m_color{1.f, 1.f, 1.f};
  bool m_visible{true};

  void dispatch();
  void detach();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Light *, ANARI_LIGHT);
