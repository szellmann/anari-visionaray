#include "Spot.h"

namespace visionaray {

Spot::Spot(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::Spot;
}

Spot::~Spot()
{
  detach();
}

void Spot::commit()
{
  Light::commit();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, -1.f));
  m_openingAngle = getParam<float>("openingAngle", M_PI);
  m_falloffAngle = getParam<float>("falloffAngle", 0.1f);
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());

  vlight.asSpot.set_position(m_position);
  vlight.asSpot.set_spot_direction(m_direction);
  vlight.asSpot.set_spot_cutoff(m_openingAngle);
  vlight.asSpot.set_spot_exponent(0.f); // TODO: compute from falloff angle
  vlight.asSpot.set_cl(m_color);
  vlight.asSpot.set_kl(m_intensity);
  vlight.asSpot.set_constant_attenuation(1.f);
  vlight.asSpot.set_linear_attenuation(0.f);
  vlight.asSpot.set_quadratic_attenuation(0.f);

  dispatch();
}

} // visionaray
