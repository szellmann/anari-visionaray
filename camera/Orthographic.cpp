
#include "Orthographic.h"

namespace visionaray {

Orthographic::Orthographic(VisionarayGlobalState *s) : Camera(s)
{
  vcam.type = dco::Camera::Ortho;
}

void Orthographic::commitParameters()
{
  Camera::commitParameters();
  m_aspect = getParam<float>("aspect", 1.f);
  m_height = getParam<float>("height", 1.f);
}

void Orthographic::finalize()
{
  Camera::finalize();
  vcam.asOrthoCam.init(m_pos, m_dir, m_up, m_aspect, m_height, m_imageRegion);
}

} // namespace visionaray
