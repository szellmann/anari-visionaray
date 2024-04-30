
// ours
#include "VisionarayScene.h"

namespace visionaray {

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
#ifdef WITH_CUDA
  , m_gpuScene(new VisionaraySceneGPU(this))
#endif
{
  this->type = type;

  if (type == World) {
    m_worldID = deviceState()->dcos.TLSs.alloc({});
    deviceState()->dcos.worldEPS.alloc(1e-3f);
    deviceState()->dcos.worlds.alloc(dco::World{});
  }
  m_groupID = deviceState()->dcos.groups.alloc(dco::Group{});
}

VisionaraySceneImpl::~VisionaraySceneImpl()
{
  if (type == World) {
    deviceState()->dcos.TLSs.free(m_worldID);
    deviceState()->dcos.worlds.free(m_worldID);
  }
  deviceState()->dcos.groups.free(m_groupID);
}

void VisionaraySceneImpl::commit()
{
#ifdef WITH_CUDA
  m_gpuScene->commit();
#else
  unsigned triangleCount = 0;
  unsigned quadCount = 0;
  unsigned sphereCount = 0;
  unsigned coneCount = 0;
  unsigned cylinderCount = 0;
  unsigned bezierCurveCount = 0;
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
      case dco::Geometry::Cone:
        coneCount++;
        break;
      case dco::Geometry::Cylinder:
        cylinderCount++;
        break;
      case dco::Geometry::BezierCurve:
        bezierCurveCount++;
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

  m_accelStorage.triangleBLSs.resize(triangleCount);
  m_accelStorage.quadBLSs.resize(quadCount);
  m_accelStorage.sphereBLSs.resize(sphereCount);
  m_accelStorage.coneBLSs.resize(coneCount);
  m_accelStorage.cylinderBLSs.resize(cylinderCount);
  m_accelStorage.bezierCurveBLSs.resize(bezierCurveCount);
  m_accelStorage.isoSurfaceBLSs.resize(isoCount);
  m_accelStorage.volumeBLSs.resize(volumeCount);
  // No instance storage: instance BLSs are the TLSs of child scenes

  // first, build BLSs
  triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                = bezierCurveCount = isoCount = volumeCount = 0;
  for (const dco::Handle &geomID : m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    binned_sah_builder builder;

    if (geom.type == dco::Geometry::Triangle) {
      unsigned index = triangleCount++;
      builder.enable_spatial_splits(true);
      m_accelStorage.triangleBLSs[index] = builder.build(
        TriangleBVH{}, geom.asTriangle.data, geom.asTriangle.len);
    } else if (geom.type == dco::Geometry::Quad) {
      unsigned index = quadCount++;
      builder.enable_spatial_splits(true);
      m_accelStorage.quadBLSs[index] = builder.build(
        TriangleBVH{}, geom.asQuad.data, geom.asQuad.len);
    } else if (geom.type == dco::Geometry::Sphere) {
      unsigned index = sphereCount++;
      builder.enable_spatial_splits(true);
      m_accelStorage.sphereBLSs[index] = builder.build(
        SphereBVH{}, geom.asSphere.data, geom.asSphere.len);
    } else if (geom.type == dco::Geometry::Cone) {
      unsigned index = coneCount++;
      builder.enable_spatial_splits(false); // no spatial splits for cones yet!
      m_accelStorage.coneBLSs[index] = builder.build(
        ConeBVH{}, geom.asCone.data, geom.asCone.len);
    } else if (geom.type == dco::Geometry::Cylinder) {
      unsigned index = cylinderCount++;
      builder.enable_spatial_splits(false); // no spatial splits for cyls yet!
      m_accelStorage.cylinderBLSs[index] = builder.build(
        CylinderBVH{}, geom.asCylinder.data, geom.asCylinder.len);
    } else if (geom.type == dco::Geometry::BezierCurve) {
      unsigned index = bezierCurveCount++;
      builder.enable_spatial_splits(false); // no spatial splits for bez. curves yet!
      m_accelStorage.bezierCurveBLSs[index] = builder.build(
        BezierCurveBVH{}, geom.asBezierCurve.data, geom.asBezierCurve.len);
    } else if (geom.type == dco::Geometry::ISOSurface) {
      unsigned index = isoCount++;
      builder.enable_spatial_splits(false); // no spatial splits for ISOs
      m_accelStorage.isoSurfaceBLSs[index] = builder.build(
        ISOSurfaceBVH{}, &geom.asISOSurface.data, 1);
    } else if (geom.type == dco::Geometry::Volume) {
      unsigned index = volumeCount++;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs
      m_accelStorage.volumeBLSs[index] = builder.build(
        VolumeBVH{}, &geom.asVolume.data, 1);
    } else if (geom.type == dco::Geometry::Instance) {
      instanceCount++;
    }
  }

  m_BLSs.clear();
  m_worldBLSs.clear();

  // now initialize BVH refs for use in shader code:
  triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                = bezierCurveCount = isoCount = volumeCount = 0;
  for (const dco::Handle &geomID : m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    if (type == World) {
      dco::WorldBLS bls;
      bls.blsID = m_worldBLSs.alloc(bls);

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        bls.type = dco::BLS::Triangle;
        bls.asTriangle = m_accelStorage.triangleBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        bls.type = dco::BLS::Quad;
        bls.asQuad = m_accelStorage.quadBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        bls.type = dco::BLS::Sphere;
        bls.asSphere = m_accelStorage.sphereBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        bls.type = dco::BLS::Cone;
        bls.asCone = m_accelStorage.coneBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
        bls.asCylinder = m_accelStorage.cylinderBLSs[index].ref();
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        bls.type = dco::BLS::BezierCurve;
        bls.asBezierCurve = m_accelStorage.bezierCurveBLSs[index].ref();
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
        bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Volume) {
        unsigned index = volumeCount++;
        bls.type = dco::BLS::Volume;
        bls.asVolume = m_accelStorage.volumeBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Instance) {
        instanceCount++;
        bls.type = dco::BLS::Instance;
        bls.asInstance = geom.asInstance.data.instBVH;
        bls.asInstance.set_inst_id(geom.asInstance.data.instID);
      }
      m_worldBLSs.update(bls.blsID, bls);
    } else {
      dco::BLS bls;
      bls.blsID = m_BLSs.alloc(bls);

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        bls.type = dco::BLS::Triangle;
        bls.asTriangle = m_accelStorage.triangleBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        bls.type = dco::BLS::Quad;
        bls.asQuad = m_accelStorage.quadBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        bls.type = dco::BLS::Sphere;
        bls.asSphere = m_accelStorage.sphereBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        bls.type = dco::BLS::Cone;
        bls.asCone = m_accelStorage.coneBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
        bls.asCylinder = m_accelStorage.cylinderBLSs[index].ref();
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        bls.type = dco::BLS::BezierCurve;
        bls.asBezierCurve = m_accelStorage.bezierCurveBLSs[index].ref();
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
        bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Volume) {
        unsigned index = volumeCount++;
        bls.type = dco::BLS::Volume;
        bls.asVolume = m_accelStorage.volumeBLSs[index].ref();
      }
      m_BLSs.update(bls.blsID, bls);
    }
  }

  // Build TLS
  lbvh_builder tlsBuilder;
  if (type == World) {
    m_worldTLS = tlsBuilder.build(
        WorldTLS{}, m_worldBLSs.hostPtr(), m_worldBLSs.size());
  } else {
    m_TLS = tlsBuilder.build(TLS{}, m_BLSs.hostPtr(), m_BLSs.size());
  }

  // World: build flat list of lights
  if (type == World) {
    m_allLights.clear();

    // world lights
    for (unsigned i=0; i<m_lights.size(); ++i)
      m_allLights.push_back(m_lights[i]);

    // instanced lights
    for (const dco::Handle &geomID : m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
      if (!geom.isValid()) continue;

      if (geom.type != dco::Geometry::Instance) continue;

      dco::Instance inst = geom.asInstance.data;
      dco::Group group = m_state->dcos.groups[inst.groupID];

      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.push_back(group.lights[i]);
    }
  }
#endif

#if 0
  std::cout << "TLS Build (groupID: "
            << m_groupID << ", worldID: " << m_worldID << ")\n";
  std::cout << "  num nodes             : " << m_TLS.num_nodes() << '\n';
  std::cout << "  root bounds           : " << m_TLS.node(0).get_bounds().min << ' '
                                            << m_TLS.node(0).get_bounds().max << '\n';
  std::cout << "  num triangle BLSs     : " << triangleCount << '\n';
  std::cout << "  num sphere BLSs       : " << sphereCount << '\n';
  std::cout << "  num cone BLSs         : " << coneCount << '\n';
  std::cout << "  num cylinder BLSs     : " << cylinderCount << '\n';
  std::cout << "  num bezier curve BLSs : " << bezierCurveCount << '\n';
  std::cout << "  num volume BLSs       : " << volumeCount << '\n';
  std::cout << "  num iso BLSs          : " << isoCount << '\n';
  std::cout << "  num instance BLSs     : " << instanceCount << '\n';
  std::cout << "  num geoms in group    : " << m_geometries.size() << '\n';
  std::cout << "  num materials in group: " << m_materials.size() << '\n';
  std::cout << "  num lights in group   : " << m_lights.size() << '\n';
#endif // WITH_CUDA

#ifdef WITH_CUDA
  m_gpuScene->dispatch();
#else
  dispatch();
#endif
}

void VisionaraySceneImpl::release()
{
  m_geometries.clear();
  m_BLSs.clear();
  m_worldBLSs.clear();
  m_accelStorage.triangleBLSs.clear();
  m_accelStorage.sphereBLSs.clear();
  m_accelStorage.cylinderBLSs.clear();
  m_accelStorage.bezierCurveBLSs.clear();
  m_accelStorage.isoSurfaceBLSs.clear();
  m_accelStorage.volumeBLSs.clear();
  m_materials.clear();
  m_lights.clear();
}

bool VisionaraySceneImpl::isValid() const
{
#ifdef WITH_CUDA
  return m_gpuScene->isValid();
#else
  if (type == World)
    return m_worldTLS.num_nodes() > 0;
  else
    return m_TLS.num_nodes() > 0;
#endif
}

aabb VisionaraySceneImpl::getBounds() const
{
#ifdef WITH_CUDA
  return m_gpuScene->getBounds();
#else
  if (type == World)
    return m_worldTLS.node(0).get_bounds();
  else
    return m_TLS.node(0).get_bounds();
#endif
}

float VisionaraySceneImpl::getWorldEPS() const
{
  aabb bounds = getBounds();
  if (bounds.empty())
    return 1e-3f;
  float3 diag = bounds.max-bounds.min;
  return fmaxf(1e-3f, length(diag) * 1e-5f);
}

void VisionaraySceneImpl::attachGeometry(
    dco::Geometry geom, unsigned geomID, unsigned userID)
{
#ifdef WITH_CUDA
  m_gpuScene->attachGeometry(geom, geomID, userID);
#else
  m_geometries.set(geomID, geom.geomID);
  m_objIds.set(geomID, userID);

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
  } else if (geom.type == dco::Geometry::Cone) {
    for (size_t i=0;i<geom.asCone.len;++i) {
      geom.asCone.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    for (size_t i=0;i<geom.asCylinder.len;++i) {
      geom.asCylinder.data[i].geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::BezierCurve) {
    for (size_t i=0;i<geom.asBezierCurve.len;++i) {
      geom.asBezierCurve.data[i].geom_id = geomID;
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
#endif
}

void VisionaraySceneImpl::attachGeometry(
    dco::Geometry geom, dco::Material mat, unsigned geomID, unsigned userID)
{
  attachGeometry(geom, geomID, userID);

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

#ifdef WITH_CUDA
cuda_index_bvh<dco::BLS>::bvh_inst VisionaraySceneImpl::instBVH(mat4x3 xfm)
{
  return m_gpuScene->instBVH(xfm);
}
#else
index_bvh<dco::BLS>::bvh_inst VisionaraySceneImpl::instBVH(mat4x3 xfm)
{
  assert(type == Group);
  return m_TLS.inst(xfm);
}
#endif

void VisionaraySceneImpl::dispatch()
{
  // Dispatch world
  if (type == World) {
    m_state->dcos.TLSs.update(m_worldID, m_worldTLS.ref());
    m_state->dcos.worldEPS.update(m_worldID, getWorldEPS());

    dco::World world; // TODO: move TLS and EPS in here!
    world.numLights = m_allLights.size();
    world.allLights = m_allLights.devicePtr();
    m_state->dcos.worlds.update(m_worldID, world);
  }

  // Dispatch group
  if (type == Group) {
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
    group.objIds = m_objIds.devicePtr();
    group.numObjIds = m_objIds.size();
    m_state->dcos.groups.update(m_groupID, group);
  }

  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.devicePtr();
  m_state->onDevice.worldEPS = m_state->dcos.worldEPS.devicePtr();
  m_state->onDevice.groups = m_state->dcos.groups.devicePtr();
  if (type == World) {
    m_state->onDevice.worlds = m_state->dcos.worlds.devicePtr();
  }
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
