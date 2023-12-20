
// ours
#include "VisionarayScene.h"

namespace visionaray {

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
{
  this->type = type;

  if (type == World) {
    m_worldID = deviceState()->dcos.TLSs.alloc(m_TLS.ref());
  }
  m_groupID = deviceState()->dcos.groups.alloc(dco::Group{});
}

VisionaraySceneImpl::~VisionaraySceneImpl()
{
  if (type == World) {
    deviceState()->dcos.TLSs.free(m_worldID);
  }
  deviceState()->dcos.groups.free(m_groupID);
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

  for (const dco::Handle &geomID : m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
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
  for (const dco::Handle &geomID : m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    dco::BLS bls;
    bls.blsID = m_BLSs.alloc(bls);

    if (geom.type == dco::Geometry::Triangle) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = triangleCount++;
      m_accelStorage.triangleBLSs[index] = builder.build(
        TriangleBVH{}, geom.asTriangle.data, geom.asTriangle.len);

      bls.type = dco::BLS::Triangle;
      bls.asTriangle = m_accelStorage.triangleBLSs[index].ref();
    } else if (geom.type == dco::Geometry::Quad) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = quadCount++;
      m_accelStorage.quadBLSs[index] = builder.build(
        TriangleBVH{}, geom.asQuad.data, geom.asQuad.len);

      bls.type = dco::BLS::Quad;
      bls.asQuad = m_accelStorage.quadBLSs[index].ref();
    } else if (geom.type == dco::Geometry::Sphere) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(true);

      unsigned index = sphereCount++;
      m_accelStorage.sphereBLSs[index] = builder.build(
        SphereBVH{}, geom.asSphere.data, geom.asSphere.len);

      bls.type = dco::BLS::Sphere;
      bls.asSphere = m_accelStorage.sphereBLSs[index].ref();
    } else if (geom.type == dco::Geometry::Cylinder) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for cyls yet!

      unsigned index = cylinderCount++;
      m_accelStorage.cylinderBLSs[index] = builder.build(
        CylinderBVH{}, geom.asCylinder.data, geom.asCylinder.len);

      bls.type = dco::BLS::Cylinder;
      bls.asCylinder = m_accelStorage.cylinderBLSs[index].ref();
    } else if (geom.type == dco::Geometry::ISOSurface) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for ISOs

      unsigned index = isoCount++;
      m_accelStorage.isoSurfaceBLSs[index] = builder.build(
        ISOSurfaceBVH{}, &geom.asISOSurface.data, 1);

      bls.type = dco::BLS::ISOSurface;
      bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].ref();
    } else if (geom.type == dco::Geometry::Volume) {
      binned_sah_builder builder;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs

      unsigned index = volumeCount++;
      m_accelStorage.volumeBLSs[index] = builder.build(
        VolumeBVH{}, &geom.asVolume.data, 1);

      bls.type = dco::BLS::Volume;
      bls.asVolume = m_accelStorage.volumeBLSs[index].ref();
    } else if (geom.type == dco::Geometry::Instance) {
      instanceCount++;
      bls.type = dco::BLS::Instance;
      bls.asInstance = geom.asInstance.data.instBVH;
      bls.asInstance.set_inst_id(geom.asInstance.data.instID);
    }

    m_BLSs.update(bls.blsID, bls);
  }

  // Build TLS
  if (1) {
    lbvh_builder tlsBuilder;
    m_TLS = tlsBuilder.build(TLS{}, m_BLSs.hostPtr(), m_BLSs.size());
  } else { // build on device
    lbvh_builder tlsBuilder;
    m_TLS = tlsBuilder.build(TLS{}, m_BLSs.devicePtr(), m_BLSs.size());
  }

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
  m_geometries.set(geomID, geom.geomID);

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
    geom.asISOSurface.data.geomID = geomID;
  } else if (geom.type == dco::Geometry::Volume) {
    geom.asVolume.data.geomID = geomID;
  }

  m_geometries.set(geomID, geom.geomID);
  deviceState()->dcos.geometries.update(geom.geomID, geom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

void VisionaraySceneImpl::attachGeometry(
    dco::Geometry geom, dco::Material mat, unsigned geomID)
{
  attachGeometry(geom, geomID);

  m_materials.set(geomID, mat.matID);
}

void VisionaraySceneImpl::updateGeometry(dco::Geometry geom)
{
  deviceState()->dcos.geometries.update(geom.geomID, geom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

void VisionaraySceneImpl::attachLight(dco::Light light, unsigned id)
{
  m_lights.set(id, light.lightID);
}

void VisionaraySceneImpl::dispatch()
{
  // Dispatch world
  if (type == World) {
    m_state->dcos.TLSs.update(m_worldID, m_TLS.ref());
  }

  // Dispatch group
  dco::Group group;
  group.groupID = m_groupID;
  group.numBLSs = m_BLSs.size();
  group.BLSs = m_BLSs.devicePtr();
  group.numGeoms = m_geometries.size();
  group.geoms = m_geometries.devicePtr();
  group.numMaterials = m_materials.size();
  group.materials = m_materials.devicePtr();
  group.numLights = m_lights.size();
  group.lights = m_lights.devicePtr();
  m_state->dcos.groups.update(m_groupID, group);

  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.devicePtr();
  m_state->onDevice.groups = m_state->dcos.groups.devicePtr();
}

VisionarayGlobalState *VisionaraySceneImpl::deviceState()
{
  return m_state;
}

VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
{
  return std::make_shared<VisionaraySceneImpl>(type, state);
}

} // namespace visionaray
