#include "Spot.h"

namespace visionaray {

Spot::Spot(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::Spot;
}

Spot::~Spot()
{
}

void Spot::commitParameters()
{
  Light::commitParameters();
  m_position = getParam<vec3>("position", vec3(0.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, -1.f));
  m_openingAngle = getParam<float>("openingAngle", M_PI);
  m_falloffAngle = getParam<float>("falloffAngle", 0.1f);
  m_intensity = std::clamp(getParam<float>("intensity", 1.f),
      0.f,
      std::numeric_limits<float>::max());
}

void Spot::finalize()
{
  Light::finalize();

  float innerAngle = m_openingAngle - 2.f * m_falloffAngle;
  if (innerAngle < 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "falloffAngle should be smaller than half of openingAngle");
  }

  vlight.asSpot.position = m_position;
  vlight.asSpot.direction = m_direction;
  vlight.asSpot.color = m_color;
  vlight.asSpot.lightIntensity = m_intensity;

  vlight.asSpot.cosOuterAngle = cosf(m_openingAngle);
  vlight.asSpot.cosInnerAngle = cosf(innerAngle);

  dispatch();
}

} // visionaray
