// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"

namespace visionaray {

Light::Light(VisionarayGlobalState *s) : Object(ANARI_LIGHT, s) {}

Light *Light::createInstance(std::string_view /*subtype*/, VisionarayGlobalState *s)
{
  return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Light *);
