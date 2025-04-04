// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
// subtypes
#include "Directional.h"
#include "HDRI.h"
#include "Point.h"
#include "Quad.h"
#include "Spot.h"

namespace visionaray {

Light::Light(VisionarayGlobalState *s) : Object(ANARI_LIGHT, s)
{
  vlight = dco::createLight();
  vlight.lightID = deviceState()->dcos.lights.alloc(vlight);
}

Light::~Light()
{
  deviceState()->dcos.lights.free(vlight.lightID);
}

void Light::commitParameters()
{
  m_color = getParam<vec3>("color", vec3(1.f, 1.f, 1.f));
  m_visible = getParam<bool>("visible", true);
}

void Light::finalize()
{
  vlight.visible = m_visible;
}

Light *Light::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "directional")
    return new Directional(s);
  else if (subtype == "point")
    return new Point(s);
  else if (subtype == "quad")
    return new QuadLight(s);
  else if (subtype == "spot")
    return new Spot(s);
  else if (subtype == "hdri")
    return new HDRI(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

void Light::dispatch()
{
  deviceState()->dcos.lights.update(vlight.lightID, vlight);
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Light *);
