// Copyright 2023-2026 Stefan Zellmann
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

  void commitParameters() override;

  dco::Material visionarayMaterial() const { return vmat; }

 protected:
  dco::Material vmat;

  void dispatch();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Material *, ANARI_MATERIAL);
