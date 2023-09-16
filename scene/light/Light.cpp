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
  else if (subtype == "hdri")
    return new HDRI(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

void Light::dispatch()
{
  if (deviceState()->dcos.lights.size() <= vlight.lightID) {
    deviceState()->dcos.lights.resize(vlight.lightID+1);
  }
  deviceState()->dcos.lights[vlight.lightID] = vlight;

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

void Light::detach()
{
  if (deviceState()->dcos.lights.size() > vlight.lightID) {
    if (deviceState()->dcos.lights[vlight.lightID].lightID == vlight.lightID) {
      deviceState()->dcos.lights.erase(
          deviceState()->dcos.lights.begin() + vlight.lightID);
    }
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Light *);
