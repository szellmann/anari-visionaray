
// ours
#include "VisionarayScene.h"

namespace visionaray {

unsigned VisionaraySceneImpl::nextWorldID = 0;
unsigned VisionaraySceneImpl::nextGroupID = 0;

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
{
  this->type = type;

  if (type == World)
    m_worldID = nextWorldID++;
  m_groupID = nextGroupID++;
}

VisionaraySceneImpl::~VisionaraySceneImpl()
{
  if (type == World)
    nextWorldID--;
  nextGroupID--;
}

void VisionaraySceneImpl::commit()
{
  unsigned triangleCount = 0;
  unsigned quadCount = 0;
  unsigned sphereCount = 0;
  unsigned cylinderCount = 0;
  unsigned isoCount = 0;
  unsigned volumeCount = 0;
  unsigned instanceCount = 0;

  for (const auto &geom : m_geometries) {
    if (!geom.isValid()) continue;
    switch (geom.type) {
      case dco::Geometry::Triangle:
        triangleCount++;
        break;
      case dco::Geometry::Quad:
        quadCount++;
        break;
      case dco::Geometry::Sphere:
        sphereCount++;
        break;
      case dco::Geometry::Cylinder:
        cylinderCount++;
        break;
      case dco::Geometry::ISOSurface:
        isoCount++;
        break;
      case dco::Geometry::Volume:
        volumeCount++;
        break;
      case dco::Geometry::Instance:
      default:
        break;
    }
  }

  m_BLSs.clear();

  m_accelStorage.triangleBLSs.resize(triangleCount);
  m_accelStorage.quadBLSs.resize(quadCount);
  m_accelStorage.sphereBLSs.resize(sphereCount);
  m_accelStorage.cylinderBLSs.resize(cylinderCount);
  m_accelStorage.isoSurfaceBLSs.resize(isoCount);
  m_accelStorage.volumeBLSs.resize(volumeCount);
  // No instance storage: instance BLSs are the TLSs of child scenes

  triangleCount = quadCount = sphereCount = cylinderCount = isoCount = volumeCount = 0;
  for (const auto &geom : m_geometries) {
    if (!geom.isValid()) continue;
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
    } else if (geom.type == dco::Geometry::Quad) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = quadCount++;
      m_accelStorage.quadBLSs[index] = builder.build(
        TriangleBVH{}, geom.asQuad.data, geom.asQuad.len);

      dco::BLS bls;
      bls.type = dco::BLS::Quad;
      bls.asQuad = m_accelStorage.quadBLSs[index].ref();
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
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for cyls yet!

      unsigned index = cylinderCount++;
      m_accelStorage.cylinderBLSs[index] = builder.build(
        CylinderBVH{}, geom.asCylinder.data, geom.asCylinder.len);

      dco::BLS bls;
      bls.type = dco::BLS::Cylinder;
      bls.asCylinder = m_accelStorage.cylinderBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == dco::Geometry::ISOSurface) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for ISOs

      unsigned index = isoCount++;
      m_accelStorage.isoSurfaceBLSs[index] = builder.build(
        ISOSurfaceBVH{}, &geom.asISOSurface.data, 1);

      dco::BLS bls;
      bls.type = dco::BLS::ISOSurface;
      bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == dco::Geometry::Volume) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs

      unsigned index = volumeCount++;
      m_accelStorage.volumeBLSs[index] = builder.build(
        VolumeBVH{}, &geom.asVolume.data, 1);

      dco::BLS bls;
      bls.type = dco::BLS::Volume;
      bls.asVolume = m_accelStorage.volumeBLSs[index].ref();
      m_BLSs.push_back(bls);
    } else if (geom.type == dco::Geometry::Instance) {
      instanceCount++;
      dco::BLS bls;
      bls.type = dco::BLS::Instance;
      mat3f rot = top_left(geom.asInstance.data.xfm);
      vec3f trans(geom.asInstance.data.xfm(0,3),
                  geom.asInstance.data.xfm(1,3),
                  geom.asInstance.data.xfm(2,3));
      mat4x3 xfm{rot, trans};
      bls.asInstance = geom.asInstance.data.scene->m_TLS.inst(xfm);
      bls.asInstance.set_inst_id(geom.asInstance.data.instID);
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
  std::cout << "  num volume BLSs  : " << volumeCount << '\n';
  std::cout << "  num iso BLSs     : " << isoCount << '\n';
  std::cout << "  num instance BLSs: " << instanceCount << '\n';
#endif

  dispatch();
}

void VisionaraySceneImpl::release()
{
//detach();

  m_geometries.clear();
  m_BLSs.clear();
  m_accelStorage.triangleBLSs.clear();
  m_accelStorage.sphereBLSs.clear();
  m_accelStorage.cylinderBLSs.clear();
  m_accelStorage.isoSurfaceBLSs.clear();
  m_accelStorage.volumeBLSs.clear();
  m_materials.clear();
  m_lights.clear();
}

void VisionaraySceneImpl::attachGeometry(dco::Geometry geom, unsigned geomID)
{
  if (m_geometries.size() <= geomID)
    m_geometries.resize(geomID+1);

  geom.geomID = geomID;

  // Patch geomID into scene primitives
  if (geom.type == dco::Geometry::Triangle) {
    for (size_t i=0;i<geom.asTriangle.len;++i) {
      geom.asTriangle.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Quad) {
    for (size_t i=0;i<geom.asQuad.len;++i) {
      geom.asQuad.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Sphere) {
    for (size_t i=0;i<geom.asSphere.len;++i) {
      geom.asSphere.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    for (size_t i=0;i<geom.asCylinder.len;++i) {
      geom.asCylinder.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::ISOSurface) {
    geom.asISOSurface.data.isoID = geomID;
  } else if (geom.type == dco::Geometry::Volume) {
    /* volumes do this themselves, on commit! */
  }

  m_geometries[geomID] = geom;
}

void VisionaraySceneImpl::attachGeometry(
    dco::Geometry geom, dco::Material mat, unsigned geomID)
{
  attachGeometry(geom, geomID);

  if (m_materials.size() <= geomID)
    m_materials.resize(geomID+1);

  m_materials[geomID] = mat;
}

void VisionaraySceneImpl::updateGeometry(dco::Geometry geom)
{
  unsigned geomID = geom.geomID;

  assert(geomID < m_geometries.size());

  m_geometries[geomID] = geom;
}

void VisionaraySceneImpl::addLight(dco::Light light)
{
  light.lightID = m_lights.alloc(light);

  m_lights.update(light.lightID, light);
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
  m_state->dcos.groups[m_groupID].numGeoms = m_geometries.size();
  m_state->dcos.groups[m_groupID].geoms = m_geometries.data();
  m_state->dcos.groups[m_groupID].numMaterials = m_materials.size();
  m_state->dcos.groups[m_groupID].materials = m_materials.data();
  m_state->dcos.groups[m_groupID].numLights = m_lights.size();
  m_state->dcos.groups[m_groupID].lights = m_lights.devicePtr();

  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.data();
  m_state->onDevice.groups = m_state->dcos.groups.data();
}

void VisionaraySceneImpl::detach()
{
  // Detach world
  if (m_state->dcos.TLSs.size() > m_worldID) {
    if (m_state->dcos.TLSs[m_worldID] == m_TLS.ref()) {
      m_state->dcos.TLSs.erase(m_state->dcos.TLSs.begin() + m_worldID);
    }
  }

  // Detach group
  if (m_state->dcos.groups.size() > m_groupID) {
    if (m_state->dcos.groups[m_groupID].groupID == m_groupID) {
      m_state->dcos.groups.erase(m_state->dcos.groups.begin() + m_groupID);
    }
  }
}

VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
{
  return std::make_shared<VisionaraySceneImpl>(type, state);
}

} // namespace visionaray
