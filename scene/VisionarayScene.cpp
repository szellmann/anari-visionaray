
// ours
#include "VisionarayScene.h"

namespace visionaray {

void VisionarayScene::commit()
{
  // std::cout << "commit " << this << '\n';
  // for (auto &g : m_geometries) {
  //   std::cout << g.type << '\n';
  // }
}

void VisionarayScene::release()
{
  m_geometries.clear();
  m_triangleBLSs.clear();
  m_materials.clear();
}

void VisionarayScene::attachGeometry(VisionarayGeometry geom, unsigned geomID)
{
  if (m_geometries.size() <= geomID)
    m_geometries.resize(geomID+1);

  m_geometries[geomID] = geom;
}

} // namespace visionaray
