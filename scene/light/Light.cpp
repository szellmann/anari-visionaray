// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
// subtypes
#include "Directional.h"
#include "HDRI.h"
#include "Point.h"

namespace visionaray {

Light::Light(VisionarayGlobalState *s) : Object(ANARI_LIGHT, s)
{
  memset(&vlight,0,sizeof(vlight));
  vlight.lightID = deviceState()->dcos.lights.alloc(vlight);
  s->objectCounts.lights++;
}

Light::~Light()
{
  detach();
  deviceState()->objectCounts.lights--;
}

void Light::commit()
{
  m_color = getParam<vec3>("color", vec3(1.f, 1.f, 1.f));
  m_visible = getParam<bool>("visible", true);

  vlight.visible = m_visible;
}

Light *Light::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "directional")
    return new Directional(s);
  else if (subtype == "point")
    return new Point(s);
  else if (subtype == "hdri")
    return new HDRI(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

void Light::dispatch()
{
  deviceState()->dcos.lights.update(vlight.lightID, vlight);

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

void Light::detach()
{
  deviceState()->dcos.lights.free(vlight.lightID);

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Light *);
