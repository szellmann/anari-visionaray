
// ours
#include "VisionarayScene.h"

namespace visionaray {

void VisionaraySceneImpl::commit()
{
  // std::cout << "commit " << this << '\n';
  // for (auto &g : m_geometries) {
  //   std::cout << g.type << '\n';
  // }
}

void VisionaraySceneImpl::release()
{
  m_geometries.clear();
  //m_triangleBLSs.clear();
  m_materials.clear();
}

void VisionaraySceneImpl::attachGeometry(VisionarayGeometry geom, unsigned geomID)
{
  if (m_geometries.size() <= geomID)
    m_geometries.resize(geomID+1);

  m_geometries[geomID] = geom;
}

VisionarayScene newVisionarayScene()
{
  return std::make_shared<VisionaraySceneImpl>();
}

} // namespace visionaray
