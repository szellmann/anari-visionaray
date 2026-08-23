// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"
#include "light/Light.h"
#include "DeviceBVH.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

struct VisionarayScene
{
  enum Type { World, Group, };
  Type type;

  // Surface data //
  DeviceHandleArray m_instances;
  DeviceHandleArray m_geometries;
  DeviceHandleArray m_materials;
  DeviceHandleArray m_volumes;
  DeviceHandleArray m_lights;
  HostDeviceArray<uint32_t> m_objIds;

  // flat list of lights (only used if type is World!)
  DeviceObjectArray<dco::LightRef> m_allLights;

  // Accels //
  DeviceBVH<dco::BLS> m_TLS;
  DeviceBVH<dco::Instance> m_worldTLS;
  DeviceObjectArray<dco::BLS> m_BLSs;
  DeviceObjectArray<dco::Instance> m_worldBLSs;

  // Internal state //
  unsigned m_worldID{UINT_MAX};
  unsigned m_groupID{UINT_MAX};
  VisionarayGlobalState *m_state{nullptr};

  // Interface //
  VisionarayScene(Type type, VisionarayGlobalState *state);
  ~VisionarayScene();
  void commit();
  void reset();
  void release();
  bool isValid() const;

  void attachInstance(dco::Instance inst, unsigned instID, unsigned userID=~0u);
  void attachSurface(dco::Surface surf, dco::BLS bls, unsigned geomID, unsigned userID=~0u);
  void attachVolume(dco::Volume vol, dco::BLS bls, unsigned geomID, unsigned userID=~0u);
  void attachLight(dco::Light light, unsigned id);
  aabb getBounds();
  bvh_ref_t<dco::BLS> refBVH();
  void copyToDevice();

 private:
  void dispatch();

  VisionarayGlobalState *deviceState();
};

} // namespace visionaray
