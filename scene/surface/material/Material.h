// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/generic_material.h"
#include "visionaray/material.h"
// ours
#include "Object.h"
#include "sampler/Sampler.h"

namespace visionaray {

typedef generic_material<matte<float>> VisionarayMaterial;

struct Material : public Object
{
  Material(VisionarayGlobalState *s);
  ~Material() override;

  static Material *createInstance(
      std::string_view subtype, VisionarayGlobalState *s);

  void commit() override;

  VisionarayMaterial visionarayMaterial() const { return vmat; }

 protected:
  VisionarayMaterial vmat;
  float4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Material *, ANARI_MATERIAL);
