// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// ours
#include "VisionarayScene.h"

namespace visionaray {

VisionarayScene::VisionarayScene(
    VisionarayScene::Type type, VisionarayGlobalState *state)
  : type(type), m_state(state), m_TLS(state), m_worldTLS(state),
    m_allLights(state->syncContext), m_BLSs(state->syncContext),
    m_worldBLSs(state->syncContext)
{
  reset();
}

VisionarayScene::~VisionarayScene()
{
  if (type == World) {
    deviceState()->dcos.TLSs.free(m_worldID);
    deviceState()->dcos.worlds.free(m_worldID);
  }
  deviceState()->dcos.groups.free(m_groupID);
}

void VisionarayScene::commit()
{
  if (type == World) {
    // Build TLS
    if (!m_worldBLSs.empty()) {
#if defined(WITH_HIP)
      m_worldTLS.update(m_worldBLSs.devicePtr(),
                        m_worldBLSs.size(),
                        0); // no device LBVH builder on hip yet!
#else
      m_worldTLS.update(m_worldBLSs.devicePtr(),
                        m_worldBLSs.size(),
                        BVH_FLAG_PREFER_FAST_BUILD | BVH_FLAG_NO_STREAM_SYNCHRONIZE);
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

      const dco::Instance &inst = deviceState()->dcos.instances[instID];

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = m_state->dcos.groups[inst.groupID];

      for (unsigned i=0; i<group.numLights; ++i)
        m_allLights.alloc({group.lights[i], inst.instID});
    }
  } else {
    // Build TLS
    if (!m_BLSs.empty()) {
      m_TLS.update(m_BLSs.devicePtr(),m_BLSs.size(),
                   BVH_FLAG_PREFER_FAST_BUILD | BVH_FLAG_NO_STREAM_SYNCHRONIZE);
    }
  }

  dispatch();
}

void VisionarayScene::release()
{
  m_instances.clear();
  m_geometries.clear();
  m_BLSs.clear();
  m_worldBLSs.clear();
  m_materials.clear();
  m_lights.clear();
}

void VisionarayScene::reset()
{
  release();

  if (type == World) {
    m_worldID = deviceState()->dcos.TLSs.alloc({});
    deviceState()->dcos.worlds.alloc(dco::createWorld());
  }
  m_groupID = deviceState()->dcos.groups.alloc(dco::createGroup());
}

bool VisionarayScene::isValid() const
{
  if (type == World)
    return m_worldTLS.lastRebuildTime() > m_worldTLS.lastUpdateTime();
  else
    return m_TLS.lastRebuildTime() > m_TLS.lastUpdateTime();
}

aabb VisionarayScene::getBounds()
{
  // bounds that were valid when commit was called:
  if (type == World)
    return m_worldTLS.getBounds();
  else
    return m_TLS.getBounds();
}

void VisionarayScene::attachInstance(dco::Instance inst, unsigned userID)
{
  size_t instID = m_instances.size();

  m_instances.set(instID, inst.instID, ~0u);
  m_objIds.set(instID, userID, ~0u); // TODO: separate inst/geom

  if (inst.theBVH.num_nodes() == 0)
    return;

  m_worldBLSs.alloc(inst);
}

void VisionarayScene::attachSurface(dco::Surface surf, dco::BLS bls, unsigned userID)
{
  if (!dco::validHandle(surf.geomID))
    return;

  size_t objID = m_geometries.size();

  dco::Geometry geom = deviceState()->dcos.geometries[surf.geomID];

  if (geom.primitives.len == 0)
    return;

  m_geometries.set(objID, geom.geomID, ~0u);
  m_objIds.set(objID, userID, ~0u);

  // That's the ID local to the group the volume is in
  // (this object):
  bls.localID = objID;
  // now add the BLS to our group:
  bls.blsID = m_BLSs.alloc(bls);

  if (!dco::validHandle(surf.matID))
    return;

  dco::Material mat = deviceState()->dcos.materials[surf.matID];

  m_materials.set(objID, mat.matID, ~0u);
}

void VisionarayScene::attachVolume(dco::Volume vol, dco::BLS bls, unsigned userID)
{
  size_t objID = m_volumes.size();

  m_volumes.set(objID, vol.volID, ~0u);
  m_objIds.set(objID, userID, ~0u);

  // That's the ID local to the group the volume is in
  // (this object):
  bls.localID = objID;
  // now add the BLS to our group:
  bls.blsID = m_BLSs.alloc(bls);

  dco::Material mat = dco::createMaterial(); // invalid!
  if (vol.gradientShading) {
    mat.type = dco::Material::Matte;
    mat.asMatte.color = dco::createMaterialParamRGB();
    mat.asMatte.color.rgb = float3(1,1,1);
  }
  mat.matID = deviceState()->dcos.materials.alloc(mat);

  m_materials.set(objID, mat.matID, ~0u);
}

void VisionarayScene::attachLight(dco::Light light, unsigned userID)
{
  size_t objID = m_lights.size();
  m_lights.set(objID, light.lightID, ~0u);
}

bvh_ref_t<dco::BLS> VisionarayScene::refBVH()
{
  assert(type == Group);
  return m_TLS.deviceBVH2();
}

void VisionarayScene::copyToDevice()
{
  // Upload/set accessible pointers
  m_state->onDevice.TLSs = m_state->dcos.TLSs.devicePtr();
  m_state->onDevice.worlds = m_state->dcos.worlds.devicePtr();
  m_state->onDevice.groups = m_state->dcos.groups.devicePtr();
  m_state->onDevice.surfaces = m_state->dcos.surfaces.devicePtr();
  m_state->onDevice.instances = m_state->dcos.instances.devicePtr();
  m_state->onDevice.geometries = m_state->dcos.geometries.devicePtr();
  m_state->onDevice.materials = m_state->dcos.materials.devicePtr();
  m_state->onDevice.samplers = m_state->dcos.samplers.devicePtr();
  m_state->onDevice.volumes = m_state->dcos.volumes.devicePtr();
  m_state->onDevice.spatialFields = m_state->dcos.spatialFields.devicePtr();
  m_state->onDevice.lights = m_state->dcos.lights.devicePtr();
}

void VisionarayScene::dispatch()
{
  // Dispatch world
  if (type == World) {
    deviceState()->dcos.TLSs.update(m_worldID, m_worldTLS.deviceBVH2());

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
}

VisionarayGlobalState *VisionarayScene::deviceState()
{
  return m_state;
}

} // namespace visionaray
