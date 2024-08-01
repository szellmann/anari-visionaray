
// ours
#include "VisionarayScene.h"

namespace visionaray {

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
#if defined(WITH_CUDA) || defined(WITH_HIP)
  , m_gpuScene(new VisionaraySceneGPU(this))
#endif
{
  this->type = type;

  if (type == World) {
    m_worldID = deviceState()->dcos.TLSs.alloc({});
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
#if defined(WITH_CUDA) || defined(WITH_HIP)
  m_gpuScene->commit();
#else

  if (type == World) {
    m_worldBLSs.clear();
    for (const dco::Handle &instID : m_instances) {
      if (!dco::validHandle(instID)) continue;

      const dco::Instance &inst = deviceState()->dcos.instances[instID];

      m_worldBLSs.alloc(inst);
    }

    // Build TLS
    lbvh_builder tlsBuilder;
    m_worldTLS = tlsBuilder.build(
        WorldTLS{}, m_worldBLSs.hostPtr(), m_worldBLSs.size());

    // Build flat list of lights
    m_allLights.clear();

    // world lights
    for (unsigned i=0; i<m_lights.size(); ++i)
      m_allLights.push_back(m_lights[i]);

    // instanced lights
    for (const dco::Handle &instID : m_instances) {
      if (!dco::validHandle(instID)) continue;

      const dco::Instance &inst = deviceState()->dcos.instances[instID];

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = m_state->dcos.groups[inst.groupID];

      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.push_back(group.lights[i]);
    }
  } else {
    unsigned triangleCount = 0;
    unsigned quadCount = 0;
    unsigned sphereCount = 0;
    unsigned coneCount = 0;
    unsigned cylinderCount = 0;
    unsigned bezierCurveCount = 0;
    unsigned isoCount = 0;
    unsigned volumeCount = 0;

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
        default:
          break;
      }
    }

    for (const dco::Handle &volID : m_volumes) {
      if (!dco::validHandle(volID)) continue;
      volumeCount++;
    }

    m_accelStorage.triangleBLSs.resize(triangleCount);
    m_accelStorage.quadBLSs.resize(quadCount);
    m_accelStorage.sphereBLSs.resize(sphereCount);
    m_accelStorage.coneBLSs.resize(coneCount);
    m_accelStorage.cylinderBLSs.resize(cylinderCount);
    m_accelStorage.bezierCurveBLSs.resize(bezierCurveCount);
    m_accelStorage.isoSurfaceBLSs.resize(isoCount);
    m_accelStorage.volumeBLSs.resize(volumeCount);

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
          TriangleBVH{}, (const dco::Triangle *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        builder.enable_spatial_splits(true);
        m_accelStorage.quadBLSs[index] = builder.build(
          TriangleBVH{}, (const dco::Triangle *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        builder.enable_spatial_splits(true);
        m_accelStorage.sphereBLSs[index] = builder.build(
          SphereBVH{}, (const dco::Sphere *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        builder.enable_spatial_splits(false); // no spatial splits for cones yet!
        m_accelStorage.coneBLSs[index] = builder.build(
          ConeBVH{}, (const dco::Cone *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        builder.enable_spatial_splits(false); // no spatial splits for cyls yet!
        m_accelStorage.cylinderBLSs[index] = builder.build(
          CylinderBVH{}, (const dco::Cylinder *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        builder.enable_spatial_splits(false); // no spatial splits for bez. curves yet!
        m_accelStorage.bezierCurveBLSs[index] = builder.build(
          BezierCurveBVH{},
          (const dco::BezierCurve *)geom.primitives.data, geom.primitives.len);
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        builder.enable_spatial_splits(false); // no spatial splits for ISOs
        m_accelStorage.isoSurfaceBLSs[index] = builder.build(
          ISOSurfaceBVH{}, (const dco::ISOSurface *)geom.primitives.data, 1);
      }
    }

    for (const dco::Handle &volID : m_volumes) {
      if (!dco::validHandle(volID)) continue;

      const dco::Volume &vol = deviceState()->dcos.volumes[volID];

      binned_sah_builder builder;
      unsigned index = volumeCount++;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs
      m_accelStorage.volumeBLSs[index] = builder.build(VolumeBVH{}, &vol, 1);
    }

    m_BLSs.clear();

    // now initialize BVH refs for use in shader code:
    triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                  = bezierCurveCount = isoCount = volumeCount = 0;
    for (const dco::Handle &geomID : m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
      if (!geom.isValid()) continue;

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
      }
      m_BLSs.update(bls.blsID, bls);
    }

    for (const dco::Handle &volID : m_volumes) {
      if (!dco::validHandle(volID)) continue;

      dco::BLS bls;
      bls.blsID = m_BLSs.alloc(bls);

      unsigned index = volumeCount++;
      bls.type = dco::BLS::Volume;
      bls.asVolume = m_accelStorage.volumeBLSs[index].ref();

      m_BLSs.update(bls.blsID, bls);
    }

    // Build TLS
    lbvh_builder tlsBuilder;
    m_TLS = tlsBuilder.build(TLS{}, m_BLSs.hostPtr(), m_BLSs.size());
  }
#endif

#if defined(WITH_CUDA) || defined(WITH_HIP)
  m_gpuScene->dispatch();
#else
  dispatch();
#endif
}

void VisionaraySceneImpl::release()
{
  m_instances.clear();
  m_geometries.clear();
  m_BLSs.clear();
  m_worldBLSs.clear();
  m_accelStorage.triangleBLSs.clear();
  m_accelStorage.sphereBLSs.clear();
  m_accelStorage.coneBLSs.clear();
  m_accelStorage.cylinderBLSs.clear();
  m_accelStorage.bezierCurveBLSs.clear();
  m_accelStorage.isoSurfaceBLSs.clear();
  m_accelStorage.volumeBLSs.clear();
  m_materials.clear();
  m_lights.clear();
}

bool VisionaraySceneImpl::isValid() const
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
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
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return m_gpuScene->getBounds();
#else
  if (type == World)
    return m_worldTLS.node(0).get_bounds();
  else
    return m_TLS.node(0).get_bounds();
#endif
}

void VisionaraySceneImpl::attachInstance(
    dco::Instance inst, unsigned instID, unsigned userID)
{
  m_instances.set(instID, inst.instID);
  m_objIds.set(instID, userID); // TODO: separate inst/geom

  m_instances.set(instID, inst.instID);
  deviceState()->dcos.instances.update(inst.instID, inst);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

void VisionaraySceneImpl::attachGeometry(
    dco::Geometry geom, unsigned geomID, unsigned userID)
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  m_gpuScene->attachGeometry(geom, geomID, userID);
#else
  m_geometries.set(geomID, geom.geomID);
  m_objIds.set(geomID, userID);

  // Patch geomID into scene primitives
  if (geom.type == dco::Geometry::Triangle) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::Triangle>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Quad) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::Triangle>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Sphere) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::Sphere>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Cone) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::Cone>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::Cylinder>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::BezierCurve) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      geom.as<dco::BezierCurve>(i).geom_id = geomID;
    }
  } else if (geom.type == dco::Geometry::ISOSurface) {
    geom.as<dco::ISOSurface>(0).isoID = geomID;
    geom.as<dco::ISOSurface>(0).geomID = geomID;
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

void VisionaraySceneImpl::attachVolume(
    dco::Volume vol, unsigned volID, unsigned userID)
{
  // Patch volID into scene primitives:
  vol.volID = volID;

  m_volumes.set(volID, vol.volID);
  m_objIds.set(volID, userID);

  // Upload/set accessible pointers
  deviceState()->onDevice.volumes = deviceState()->dcos.volumes.devicePtr();
}

void VisionaraySceneImpl::updateGeometry(dco::Geometry geom)
{
  deviceState()->dcos.geometries.update(geom.geomID, geom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

void VisionaraySceneImpl::updateVolume(dco::Volume vol)
{
  deviceState()->dcos.volumes.update(vol.volID, vol);

  // Upload/set accessible pointers
  deviceState()->onDevice.volumes = deviceState()->dcos.volumes.devicePtr();
}

void VisionaraySceneImpl::attachLight(dco::Light light, unsigned id)
{
  m_lights.set(id, light.lightID);
}

#ifdef WITH_CUDA
cuda_index_bvh<dco::BLS>::bvh_ref VisionaraySceneImpl::refBVH()
{
  return m_gpuScene->refBVH();
}
#elif defined(WITH_HIP)
hip_index_bvh<dco::BLS>::bvh_ref VisionaraySceneImpl::refBVH()
{
  return m_gpuScene->refBVH();
}
#else
index_bvh<dco::BLS>::bvh_ref VisionaraySceneImpl::refBVH()
{
  assert(type == Group);
  return m_TLS.ref();
}
#endif

void VisionaraySceneImpl::dispatch()
{
  // Dispatch world
  if (type == World) {
    m_state->dcos.TLSs.update(m_worldID, m_worldTLS.ref());

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
    group.numVolumes = m_volumes.size();
    group.volumes = m_volumes.devicePtr();
    group.numLights = m_lights.size();
    group.lights = m_lights.devicePtr();
    group.objIds = m_objIds.devicePtr();
    group.numObjIds = m_objIds.size();
    m_state->dcos.groups.update(m_groupID, group);
  }

  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.devicePtr();
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
