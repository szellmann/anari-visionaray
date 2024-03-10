
#include "Orthographic.h"

namespace visionaray {

Orthographic::Orthographic(VisionarayGlobalState *s) : Camera(s) {}

void Orthographic::commit()
{
  Camera::commit();

  float aspect = getParam<float>("aspect", 1.f);
  float height = getParam<float>("height", 1.f);

  vcam.type = dco::Camera::Ortho;
  vcam.asOrthoCam.init(m_pos, m_dir, m_up, aspect, height, m_imageRegion);
}

} // namespace visionaray
