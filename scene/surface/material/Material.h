// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "DeviceCopyableObjects.h"
#include "Object.h"
#include "sampler/Sampler.h"

namespace visionaray {

struct Material : public Object
{
  Material(VisionarayGlobalState *s);
  ~Material() override;

  static Material *createInstance(
      std::string_view subtype, VisionarayGlobalState *s);

  void commit() override;

  dco::Material visionarayMaterial() const { return vmat; }

 protected:
  dco::Material vmat;
  float4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
  dco::Attribute m_colorAttribute;

};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Material *, ANARI_MATERIAL);
