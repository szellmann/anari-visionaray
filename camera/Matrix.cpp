
#include "Matrix.h"

namespace visionaray {

Matrix::Matrix(VisionarayGlobalState *s) : Camera(s) {}

void Matrix::commit()
{
  Camera::commit();

  m_proj = getParam<mat4>("proj", mat4::identity());
  m_view = getParam<mat4>("view", mat4::identity());

  vcam.type = dco::Camera::Matrix;
  vcam.asMatrixCam.set_proj_matrix(m_proj);
  vcam.asMatrixCam.set_view_matrix(m_view);
}

} // namespace visionaray
