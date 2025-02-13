
#include "Matrix.h"

namespace visionaray {

Matrix::Matrix(VisionarayGlobalState *s) : Camera(s)
{
  vcam.type = dco::Camera::Matrix;
}

void Matrix::commitParameters()
{
  Camera::commitParameters();
  m_proj = getParam<mat4>("proj", mat4::identity());
  m_view = getParam<mat4>("view", mat4::identity());
}

void Matrix::finalize()
{
  Camera::finalize();
  vcam.asMatrixCam.set_proj_matrix(m_proj);
  vcam.asMatrixCam.set_view_matrix(m_view);
}

} // namespace visionaray
