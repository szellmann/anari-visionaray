
// ours
#include "VisionarayScene.h"

namespace visionaray {

void VisionaraySceneImpl::commit()
{
  unsigned triangleCount = 0;
  unsigned sphereCount = 0;
  unsigned cylinderCount = 0;
  unsigned instanceCount = 0;

  for (const auto &geom : m_geometries) {
    switch (geom.type) {
      case VisionarayGeometry::Triangle:
        triangleCount++;
        break;
      case VisionarayGeometry::Sphere:
        sphereCount++;
        break;
      case VisionarayGeometry::Cylinder:
        cylinderCount++;
        break;
      case VisionarayGeometry::Instance:
      default:
        break;
    }
  }

  m_accelStorage.triangleBLSs.resize(triangleCount);
  m_accelStorage.sphereBLSs.resize(sphereCount);
  m_accelStorage.cylinderBLSs.resize(cylinderCount);
  // No instance storage: instance BLSs are the TLSs of child scenes

  triangleCount = sphereCount = cylinderCount = 0;
  for (const auto &geom : m_geometries) {
    if (geom.type == VisionarayGeometry::Triangle) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = triangleCount++;
      m_accelStorage.triangleBLSs[index] = builder.build(
        TriangleBVH{}, geom.asTriangle.data, geom.asTriangle.len);

      BLS bls;
      bls.type = BLS::Triangle;
      bls.asTriangle = m_accelStorage.triangleBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == VisionarayGeometry::Sphere) {
      // TODO (equiv.)
    } else if (geom.type == VisionarayGeometry::Cylinder) {

    } else if (geom.type == VisionarayGeometry::Instance) {
      instanceCount++;
      BLS bls;
      bls.type = BLS::Instance;
      mat3f rot = top_left(geom.asInstance.xfm);
      vec3f trans(geom.asInstance.xfm(0,3),
                  geom.asInstance.xfm(1,3),
                  geom.asInstance.xfm(2,3));
      mat4x3 xfm{rot, trans};
      bls.asInstance = geom.asInstance.scene->m_TLS.inst(xfm);
      m_BLSs.push_back(bls);
    }
  }

  // Build TLS
  lbvh_builder tlsBuilder;
  m_TLS = tlsBuilder.build(TLS{}, m_BLSs.data(), m_BLSs.size());

#if 0
  std::cout << "TLS built\n";
  std::cout << "  num nodes: " << m_TLS.num_nodes() << '\n';
  std::cout << "  root bounds: " << m_TLS.node(0).get_bounds().min << ' '
                                 << m_TLS.node(0).get_bounds().max << '\n';
  std::cout << "  num triangle BLSs: " << triangleCount << '\n';
  std::cout << "  num sphere BLSs  : " << sphereCount << '\n';
  std::cout << "  num cylinder BLSs: " << cylinderCount << '\n';
  std::cout << "  num instance BLSs: " << instanceCount << '\n';
#endif

  onDevice.theTLS = m_TLS.ref();
  onDevice.geoms = m_geometries.data();
}

void VisionaraySceneImpl::release()
{
  m_geometries.clear();
  m_accelStorage.triangleBLSs.clear();
  m_accelStorage.sphereBLSs.clear();
  m_accelStorage.cylinderBLSs.clear();
  m_materials.clear();
}

void VisionaraySceneImpl::attachGeometry(VisionarayGeometry geom, unsigned geomID)
{
  if (m_geometries.size() <= geomID)
    m_geometries.resize(geomID+1);

  // Patch geomID into scene primitives
  if (geom.type == VisionarayGeometry::Triangle) {
    for (size_t i=0;i<geom.asTriangle.len;++i) {
      geom.asTriangle.data[i].geom_id = geomID;
    }
  }

  m_geometries[geomID] = geom;
}

VisionarayScene newVisionarayScene()
{
  return std::make_shared<VisionaraySceneImpl>();
}

} // namespace visionaray
