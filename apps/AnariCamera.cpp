#include <anari/anari_cpp/ext/linalg.h>
#include "AnariCamera.h"
#include <iostream>

AnariCamera::AnariCamera(anari::Device device) : anariDevice(device)
{
  anariCamera = anari::newObject<anari::Camera>(anariDevice, "perspective");
  set_lens_radius(0.f);
  set_focal_distance(1.f);
}

AnariCamera::~AnariCamera()
{
  anari::release(anariDevice, anariCamera);
}

void AnariCamera::viewAll(std::array<anari::math::float3, 2> bounds)
{
  visionaray::aabb vbounds{{bounds[0][0], bounds[0][1], bounds[0][2]},
                           {bounds[1][0], bounds[1][1], bounds[1][2]}};
  view_all(vbounds);
}

void AnariCamera::commit()
{
  anari::math::float3 position(eye().x, eye().y, eye().z);
  anari::math::float3 direction
      = anari::math::float3(center().x, center().y, center().z) - position;
  anari::math::float3 up(this->up().x, this->up().y, this->up().z);
  anari::setParameter(anariDevice, anariCamera, "aspect", aspect());
  anari::setParameter(anariDevice, anariCamera, "fovy", fovy());
  anari::setParameter(anariDevice, anariCamera, "position", position);
  anari::setParameter(anariDevice, anariCamera, "direction", direction);
  anari::setParameter(anariDevice, anariCamera, "up", up);
  anari::math::float2 shutter(0.f, 1.f);
  anariSetParameter(anariDevice, anariCamera, "shutter", ANARI_FLOAT32_BOX1, &shutter);
  anari::setParameter(anariDevice, anariCamera, "apertureRadius", .1f);
  anari::setParameter(anariDevice, anariCamera, "focusDistance",
      fmaxf(1e-2f, length(direction)));
  anari::commitParameters(anariDevice, anariCamera);
}
