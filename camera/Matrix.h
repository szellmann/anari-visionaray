
#pragma once

#include "Camera.h"

namespace visionaray {

struct Matrix : public Camera
{
  Matrix(VisionarayGlobalState *s);

  void commit() override;

 private:
  mat4 m_proj;
  mat4 m_view;
};

} // namespace visionaray
