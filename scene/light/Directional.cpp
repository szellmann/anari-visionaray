#include "Directional.h"

namespace visionaray {

Directional::Directional(VisionarayGlobalState *d) : Light(d)
{
  vlight.type = dco::Light::Directional;
}

Directional::~Directional()
{
  detach();
}

void Directional::commit()
{
  Light::commit();
  m_direction =
      normalize(getParam<vec3>("direction", vec3(0.f, 0.f, -1.f)));
  m_irradiance = std::clamp(getParam<float>("irradiance", 1.f),
      0.f,
      std::numeric_limits<float>::max());
  dispatch();
}

void Directional::dispatch()
{
  if (deviceState()->dcos.lights.size() <= vlight.lightID) {
    deviceState()->dcos.lights.resize(vlight.lightID+1);
  }
  deviceState()->dcos.lights[vlight.lightID].lightID = vlight.lightID;
  deviceState()->dcos.lights[vlight.lightID].asDirectional.set_direction(-m_direction);
  deviceState()->dcos.lights[vlight.lightID].asDirectional.set_cl(m_color);
  deviceState()->dcos.lights[vlight.lightID].asDirectional.set_kl(m_irradiance); // TODO!
  deviceState()->dcos.lights[vlight.lightID].asDirectional.set_angular_diameter(15.f); // TODO!

  // Upload/set accessible pointers
  deviceState()->onDevice.lights = deviceState()->dcos.lights.data();
}

void Directional::detach()
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
