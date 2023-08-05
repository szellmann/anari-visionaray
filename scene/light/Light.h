// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

struct Light : public Object
{
  Light(VisionarayGlobalState *d);
  static Light *createInstance(std::string_view subtype, VisionarayGlobalState *d);
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Light *, ANARI_LIGHT);
