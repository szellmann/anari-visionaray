// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct Light : public Object
{
  Light(VisionarayGlobalState *d);
  virtual ~Light();

  virtual void commitParameters() override;
  virtual void finalize() override;

  static Light *createInstance(std::string_view subtype, VisionarayGlobalState *d);

  dco::Light visionarayLight() const { return vlight; }

 protected:
  dco::Light vlight;
  vec3 m_color{1.f, 1.f, 1.f};
  bool m_visible{true};

  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Light *, ANARI_LIGHT);
