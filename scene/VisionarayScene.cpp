
// ours
#include "VisionarayScene.h"

namespace visionaray {

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
{
  static unsigned nextWorldID = 0;
  static unsigned nextGroupID = 0;

  this->type = type;

  if (type == World)
    m_worldID = nextWorldID++;
  m_groupID = nextGroupID++;
}

void VisionaraySceneImpl::commit()
{
  unsigned triangleCount = 0;
  unsigned sphereCount = 0;
  unsigned cylinderCount = 0;
  unsigned instanceCount = 0;

  for (const auto &geom : m_geometries) {
    switch (geom.type) {
      case dco::Geometry::Triangle:
        triangleCount++;
        break;
      case dco::Geometry::Sphere:
        sphereCount++;
        break;
      case dco::Geometry::Cylinder:
        cylinderCount++;
        break;
      case dco::Geometry::Instance:
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
    if (geom.type == dco::Geometry::Triangle) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = triangleCount++;
      m_accelStorage.triangleBLSs[index] = builder.build(
        TriangleBVH{}, geom.asTriangle.data, geom.asTriangle.len);

      dco::BLS bls;
      bls.type = dco::BLS::Triangle;
      bls.asTriangle = m_accelStorage.triangleBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == dco::Geometry::Sphere) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = sphereCount++;
      m_accelStorage.sphereBLSs[index] = builder.build(
        SphereBVH{}, geom.asSphere.data, geom.asSphere.len);

      dco::BLS bls;
      bls.type = dco::BLS::Sphere;
      bls.asSphere = m_accelStorage.sphereBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == dco::Geometry::Cylinder) {
      // TODO (equiv.)
    } else if (geom.type == dco::Geometry::Instance) {
      instanceCount++;
      dco::BLS bls;
      bls.type = dco::BLS::Instance;
      mat3f rot = top_left(geom.asInstance.xfm);
      vec3f trans(geom.asInstance.xfm(0,3),
                  geom.asInstance.xfm(1,3),
                  geom.asInstance.xfm(2,3));
      mat4x3 xfm{rot, trans};
      bls.asInstance = geom.asInstance.scene->m_TLS.inst(xfm);
      bls.asInstance.set_inst_id(geom.asInstance.instID);
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

  dispatch();
}

void VisionaraySceneImpl::release()
{
  m_geometries.clear();
  m_BLSs.clear();
  m_accelStorage.triangleBLSs.clear();
  m_accelStorage.sphereBLSs.clear();
  m_accelStorage.cylinderBLSs.clear();
  m_materials.clear();
}

void VisionaraySceneImpl::attachGeometry(dco::Geometry geom, unsigned geomID)
{
  if (m_geometries.size() <= geomID)
    m_geometries.resize(geomID+1);

  // Patch geomID into scene primitives
  if (geom.type == dco::Geometry::Triangle) {
    for (size_t i=0;i<geom.asTriangle.len;++i) {
      geom.asTriangle.data[i].geom_id = geomID;
    }
  }

  m_geometries[geomID] = geom;
}

void VisionaraySceneImpl::dispatch()
{
  // Dispatch world
  if (m_worldID < UINT_MAX) {
    if (m_state->dcos.TLSs.size() <= m_worldID) {
      m_state->dcos.TLSs.resize(m_worldID+1);
    }
    m_state->dcos.TLSs[m_worldID] = m_TLS.ref();
  }

  // Dispatch group
  if (m_state->dcos.groups.size() <= m_groupID) {
    m_state->dcos.groups.resize(m_groupID+1);
  }
  m_state->dcos.groups[m_groupID].groupID = m_groupID;
  m_state->dcos.groups[m_groupID].geoms = m_geometries.data();

  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.data();
  m_state->onDevice.groups = m_state->dcos.groups.data();
}

VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
{
  return std::make_shared<VisionaraySceneImpl>(type, state);
}

} // namespace visionaray
