// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
// subtypes
#include "Directional.h"
#include "Point.h"

namespace visionaray {

Light::Light(VisionarayGlobalState *s) : Object(ANARI_LIGHT, s)
{
  memset(&vlight,0,sizeof(vlight));
  vlight.lightID = s->objectCounts.lights++;
}

Light::~Light()
{
  deviceState()->objectCounts.lights--;
}

void Light::commit()
{
  m_color = getParam<vec3>("color", vec3(1.f, 1.f, 1.f));
}

Light *Light::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "directional")
    return new Directional(s);
  else if (subtype == "point")
    return new Point(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Light *);
