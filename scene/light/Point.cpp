#include "Point.h"

namespace visionaray {

Point::Point(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::Point;
}

Point::~Point()
{
  detach();
}

void Point::commit()
{
  Light::commit();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, -1.f));
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());
  dispatch();
}

void Point::dispatch()
{
  if (deviceState()->dcos.lights.size() <= vlight.lightID) {
    deviceState()->dcos.lights.resize(vlight.lightID+1);
  }
  deviceState()->dcos.lights[vlight.lightID].lightID = vlight.lightID;
  deviceState()->dcos.lights[vlight.lightID].type = vlight.type;
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_position(m_position);
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_cl(m_color);
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_kl(m_intensity);
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_constant_attenuation(1.f);
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_linear_attenuation(0.f);
  deviceState()->dcos.lights[vlight.lightID].asPoint.set_quadratic_attenuation(0.f);

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

void Point::detach()
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

} // visionaray
