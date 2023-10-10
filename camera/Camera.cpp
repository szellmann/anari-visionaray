
#include "Camera.h"
// specific types
#include "Matrix.h"
#include "Perspective.h"

namespace visionaray {

Camera::Camera(VisionarayGlobalState *s) : Object(ANARI_CAMERA, s)
{
  s->objectCounts.cameras++;
}

Camera::~Camera()
{
  deviceState()->objectCounts.cameras--;
}

Camera *Camera::createInstance(std::string_view type, VisionarayGlobalState *s)
{
  if (type == "matrix")
    return new Matrix(s);
  else if (type == "perspective")
    return new Perspective(s);
  else
    return (Camera *)new UnknownObject(ANARI_CAMERA, s);
}

void Camera::commit()
{
  m_pos = getParam<visionaray::vec3f>("position", visionaray::vec3f(0.f));
  m_dir = normalize(getParam<visionaray::vec3f>("direction", visionaray::vec3f(0.f, 0.f, 1.f)));
  m_up = normalize(getParam<visionaray::vec3f>("up", visionaray::vec3f(0.f, 1.f, 0.f)));
  m_imageRegion = getParam<box2f>("imageRegion", box2f{{0.f,0.f}, {1.f,1.f}});
  markUpdated();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Camera *);
