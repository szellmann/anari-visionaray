// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// ours
#include "VisionarayScene.h"

namespace visionaray {
namespace dco {
template<typename P> VSNRAY_FUNC
inline aabb get_prim_bounds(const P &p)
{ return get_bounds(p); }
} // namespace dco

#if defined(WITH_CUDA) || defined(WITH_HIP)
template <typename Obj>
__global__ static void getPrimBoundsGPU(Obj obj, aabb *bounds)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  *bounds = get_prim_bounds(obj);
}
#endif

template <typename Obj>
static aabb getPrimBounds(const Obj &obj)
{
#if defined(WITH_CUDA)
  aabb *bounds;
  CUDA_SAFE_CALL(cudaMalloc(&bounds, sizeof(aabb)));
  getPrimBoundsGPU<<<1,1>>>(obj, bounds);
  aabb hostBounds;
  CUDA_SAFE_CALL(
      cudaMemcpy(&hostBounds, bounds, sizeof(aabb), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(bounds));
  return hostBounds;
#elif defined(WITH_HIP)
  aabb *bounds;
  HIP_SAFE_CALL(hipMalloc(&bounds, sizeof(aabb)));
  getPrimBoundsGPU<<<1,1>>>(obj, bounds);
  aabb hostBounds;
  HIP_SAFE_CALL(
      hipMemcpy(&hostBounds, bounds, sizeof(aabb), hipMemcpyDeviceToHost));
  HIP_SAFE_CALL(hipFree(bounds));
  return hostBounds;
#else
  return get_prim_bounds(obj);
#endif
}

VisionaraySceneImpl::VisionaraySceneImpl(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state)
  : m_state(state)
{
  this->type = type;

  m_bounds[0].invalidate();
  m_bounds[1].invalidate();

  if (type == World) {
    m_worldID = deviceState()->dcos.TLSs.alloc({});
    deviceState()->dcos.worlds.alloc(dco::createWorld());
  }
  m_groupID = deviceState()->dcos.groups.alloc(dco::createGroup());
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
  boundsID = !boundsID;
  m_bounds[boundsID] = m_bounds[!boundsID];
  m_bounds[!boundsID].invalidate();

  if (type == World) {
    m_worldBLSs.clear();
    for (const dco::Handle &instID : m_instances) {
      if (!dco::validHandle(instID)) continue;

#if defined(WITH_CUDA)
      dco::Instance inst;
      CUDA_SAFE_CALL(cudaMemcpy(&inst, deviceState()->onDevice.instances+instID,
                                sizeof(inst), cudaMemcpyDefault));
#elif defined(WITH_HIP)
      HIP_SAFE_CALL(hipMemcpy(&inst, deviceState()->onDevice.instances+instID,
                              sizeof(inst), hipMemcpyDefault));
#else
      const dco::Instance &inst = deviceState()->dcos.instances[instID];
#endif
      if (inst.theBVH.num_nodes() == 0) continue;

      m_worldBLSs.alloc(inst);
    }

    // Build TLS
    if (!m_worldBLSs.empty()) {
#if defined(WITH_HIP)
      m_worldTLS.update(m_worldBLSs.hostPtr(),
                        m_worldBLSs.size(),
                        &deviceState()->threadPool,
                        0); // no device LBVH builder on hip yet!
#else
      m_worldTLS.update(m_worldBLSs.hostPtr(),
                        m_worldBLSs.size(),
                        &deviceState()->threadPool,
                        BVH_FLAG_PREFER_FAST_BUILD);
#endif
    }

    // Build flat list of lights
    m_allLights.clear();

    // world lights
    for (unsigned i=0; i<m_lights.size(); ++i)
      m_allLights.alloc({m_lights[i], ~0u});

    // instanced lights
    for (const dco::Handle &instID : m_instances) {
      if (!dco::validHandle(instID)) continue;

#if defined(WITH_CUDA)
      dco::Instance inst;
      CUDA_SAFE_CALL(cudaMemcpy(&inst, deviceState()->onDevice.instances+instID,
                                sizeof(inst), cudaMemcpyDefault));

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = deviceState()->dcos.groups[inst.groupID];

      std::vector<dco::Handle> groupLights(group.numLights);
      CUDA_SAFE_CALL(cudaMemcpy(groupLights.data(), group.lights,
                                group.numLights*sizeof(dco::Handle),
                                cudaMemcpyDefault));
      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.alloc({groupLights[i], inst.instID});
#elif defined(WITH_HIP)
      dco::Instance inst;
      HIP_SAFE_CALL(hipMemcpy(&inst, deviceState()->onDevice.instances+instID,
                              sizeof(inst), hipMemcpyDefault));

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = deviceState()->dcos.groups[inst.groupID];

      std::vector<dco::Handle> groupLights(group.numLights);
      HIP_SAFE_CALL(hipMemcpy(groupLights.data(), group.lights,
                                group.numLights*sizeof(dco::Handle),
                                hipMemcpyDefault));
      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.alloc({groupLights[i], inst.instID});
#else
      const dco::Instance &inst = deviceState()->dcos.instances[instID];

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = m_state->dcos.groups[inst.groupID];

      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.alloc({group.lights[i], inst.instID});
#endif
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

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        m_accelStorage.triangleBLSs[index].update((const dco::Triangle *)geom.primitives.data,
                                                  geom.primitives.len,
                                                  &deviceState()->threadPool,
                                                  BVH_FLAG_ENABLE_SPATIAL_SPLITS);
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        m_accelStorage.quadBLSs[index].update((const dco::Triangle *)geom.primitives.data,
                                              geom.primitives.len,
                                              &deviceState()->threadPool,
                                              BVH_FLAG_ENABLE_SPATIAL_SPLITS);
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        m_accelStorage.sphereBLSs[index].update((const dco::Sphere *)geom.primitives.data,
                                                geom.primitives.len,
                                                &deviceState()->threadPool,
                                                BVH_FLAG_ENABLE_SPATIAL_SPLITS);
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        m_accelStorage.coneBLSs[index].update((const dco::Cone *)geom.primitives.data,
                                              geom.primitives.len,
                                              &deviceState()->threadPool,
                                              0); // no spatial splits for cones yet!
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        m_accelStorage.cylinderBLSs[index].update((const dco::Cylinder *)geom.primitives.data,
                                                  geom.primitives.len,
                                                  &deviceState()->threadPool,
                                                  0); // no spatial splits for cyls yet!
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        m_accelStorage.bezierCurveBLSs[index].update((const dco::BezierCurve *)geom.primitives.data,
                                                     geom.primitives.len,
                                                     &deviceState()->threadPool,
                                                     0); // no spatial splits for bez. curves yet!
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        m_accelStorage.isoSurfaceBLSs[index].update((const dco::ISOSurface *)geom.primitives.data,
                                                     geom.primitives.len,
                                                     &deviceState()->threadPool,
                                                     0); // no spatial splits for ISOs
      }
    }

    for (const dco::Handle &volID : m_volumes) {
      if (!dco::validHandle(volID)) continue;

      const dco::Volume &vol = deviceState()->dcos.volumes[volID];

      unsigned index = volumeCount++;
      m_accelStorage.volumeBLSs[index].update(&vol, 1,
                                              &deviceState()->threadPool,
                                              0); // no spatial splits for volumes/aabbs
    }

    m_BLSs.clear();

    // now initialize BVH refs for use in shader code:
    triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                  = bezierCurveCount = isoCount = volumeCount = 0;
    for (const dco::Handle &geomID : m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];

      dco::BLS bls;
      bls.blsID = m_BLSs.alloc(bls);
      bls.localID = m_localIDs.surf[geomID];

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        bls.type = dco::BLS::Triangle;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asTriangle = m_accelStorage.triangleBLSs[index].deviceIndexBVH2();
#else
        bls.asTriangle = m_accelStorage.triangleBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        bls.type = dco::BLS::Quad;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asQuad = m_accelStorage.quadBLSs[index].deviceIndexBVH2();
#else
        bls.asQuad = m_accelStorage.quadBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        bls.type = dco::BLS::Sphere;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asSphere = m_accelStorage.sphereBLSs[index].deviceIndexBVH2();
#else
        bls.asSphere = m_accelStorage.sphereBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        bls.type = dco::BLS::Cone;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asCone = m_accelStorage.coneBLSs[index].deviceIndexBVH2();
#else
        bls.asCone = m_accelStorage.coneBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
#if defined(WITH_CUDA) || defined(WITH_HIP)

        bls.asCylinder = m_accelStorage.cylinderBLSs[index].deviceIndexBVH2();
#else
        bls.asCylinder = m_accelStorage.cylinderBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        bls.type = dco::BLS::BezierCurve;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asBezierCurve = m_accelStorage.bezierCurveBLSs[index].deviceIndexBVH2();
#else
        bls.asBezierCurve = m_accelStorage.bezierCurveBLSs[index].deviceBVH4();
#endif
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
#if defined(WITH_CUDA) || defined(WITH_HIP)
        bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].deviceIndexBVH2();
#else
        bls.asISOSurface = m_accelStorage.isoSurfaceBLSs[index].deviceBVH4();
#endif
      }
      m_BLSs.update(bls.blsID, bls);
    }

    for (const dco::Handle &volID : m_volumes) {
      if (!dco::validHandle(volID)) continue;

      dco::BLS bls;
      bls.blsID = m_BLSs.alloc(bls);
      bls.localID = m_localIDs.vol[volID];

      unsigned index = volumeCount++;
      bls.type = dco::BLS::Volume;
#if defined(WITH_CUDA) || defined(WITH_HIP)
      bls.asVolume = m_accelStorage.volumeBLSs[index].deviceIndexBVH2();
#else
      bls.asVolume = m_accelStorage.volumeBLSs[index].deviceBVH4();
#endif

      m_BLSs.update(bls.blsID, bls);
    }

    // Build TLS
    if (!m_BLSs.empty()) {
      m_TLS.update(m_BLSs.hostPtr(),m_BLSs.size(),
                   &deviceState()->threadPool,
                   BVH_FLAG_PREFER_FAST_BUILD);
    }
  }

  dispatch();
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
  if (type == World)
    return m_worldTLS.lastRebuildTime() > m_worldTLS.lastUpdateTime();
  else
    return m_TLS.lastRebuildTime() > m_TLS.lastUpdateTime();
}

aabb VisionaraySceneImpl::getBounds() const
{
  // bounds that were valid when commit was called:
  return m_bounds[boundsID];
}

void VisionaraySceneImpl::attachInstance(
    dco::Instance inst, unsigned instID, unsigned userID)
{
  m_bounds[boundsID].insert(getPrimBounds(inst));

  m_instances.set(instID, inst.instID);
  m_objIds.set(instID, userID); // TODO: separate inst/geom

  m_instances.set(instID, inst.instID);
  deviceState()->dcos.instances.update(inst.instID, inst);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

void VisionaraySceneImpl::attachSurface(
    dco::Surface surf, unsigned surfID, unsigned userID)
{
  if (!dco::validHandle(surf.geomID))
    return;

  dco::Geometry geom = deviceState()->dcos.geometries[surf.geomID];

  if (geom.primitives.len == 0)
    return;

  m_bounds[boundsID].insert(getPrimBounds(geom));

  m_geometries.set(surfID, geom.geomID);
  m_objIds.set(surfID, userID);

  if (geom.geomID >= m_localIDs.surf.size())
    m_localIDs.surf.resize(geom.geomID+1);
  m_localIDs.surf[geom.geomID] = surfID;

  if (!dco::validHandle(surf.matID))
    return;

  dco::Material mat = deviceState()->dcos.materials[surf.matID];

  m_materials.set(surfID, mat.matID);
}

void VisionaraySceneImpl::attachVolume(
    dco::Volume vol, unsigned volID, unsigned userID)
{
  // use bounds member, that way we don't need to reach for the GPU:
  m_bounds[boundsID].insert(vol.bounds);

  m_volumes.set(volID, vol.volID);
  m_objIds.set(volID, userID);

  // That's the ID local to the group the volume is in
  // (this object):
  if (vol.volID >= m_localIDs.vol.size())
    m_localIDs.vol.resize(vol.volID+1);
  m_localIDs.vol[vol.volID] = volID;
}

void VisionaraySceneImpl::attachLight(dco::Light light, unsigned id)
{
  m_lights.set(id, light.lightID);
}

index_bvh_ref_t<dco::BLS> VisionaraySceneImpl::refBVH()
{
  assert(type == Group);
  return m_TLS.deviceIndexBVH2();
}

void VisionaraySceneImpl::dispatch()
{
  // Dispatch world
  if (type == World) {
    deviceState()->dcos.TLSs.update(m_worldID, m_worldTLS.deviceIndexBVH2());

    dco::World world = dco::createWorld(); // TODO: move TLS and EPS in here!
    world.numLights = m_allLights.size();
    world.allLights = m_allLights.devicePtr();
    m_state->dcos.worlds.update(m_worldID, world);
  }

  // Dispatch group
  if (type == Group) {
    dco::Group group = dco::createGroup();
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
