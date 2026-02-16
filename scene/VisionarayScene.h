// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <memory>
// visionaray
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"
#include "light/Light.h"
#include "DeviceBVH.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

struct VisionaraySceneImpl
{
  typedef DeviceBVH<basic_triangle<3,float>> TriangleBVH;
  typedef DeviceBVH<basic_triangle<3,float>> QuadBVH;
  typedef DeviceBVH<basic_sphere<float>>     SphereBVH;
  typedef DeviceBVH<dco::Cone>               ConeBVH;
  typedef DeviceBVH<basic_cylinder<float>>   CylinderBVH;
  typedef DeviceBVH<dco::BezierCurve>        BezierCurveBVH;
  typedef DeviceBVH<dco::ISOSurface>         ISOSurfaceBVH;
  typedef DeviceBVH<dco::Volume>             VolumeBVH;

  typedef DeviceBVH<dco::BLS>                TLS;
  typedef DeviceBVH<dco::Instance>           WorldTLS;

  enum Type { World, Group, };
  Type type;

  // double-buffered bounds, attaching geoms, vols, etc. updates
  // back-buffer, buffers get swapped on commit
  aabb m_bounds[2];
  int boundsID{0};

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
  TLS m_TLS;
  WorldTLS m_worldTLS;
  DeviceObjectArray<dco::BLS> m_BLSs;
  DeviceObjectArray<dco::Instance> m_worldBLSs;

  // Accel storage //
  struct {
    std::vector<TriangleBVH>    triangleBLSs;
    std::vector<QuadBVH>        quadBLSs;
    std::vector<SphereBVH>      sphereBLSs;
    std::vector<ConeBVH>        coneBLSs;
    std::vector<CylinderBVH>    cylinderBLSs;
    std::vector<BezierCurveBVH> bezierCurveBLSs;
    std::vector<ISOSurfaceBVH>  isoSurfaceBLSs;
    std::vector<VolumeBVH>      volumeBLSs;
  } m_accelStorage;

  // Internal state //
  unsigned m_worldID{UINT_MAX};
  unsigned m_groupID{UINT_MAX};
  VisionarayGlobalState *m_state{nullptr};

  // Interface //
  VisionaraySceneImpl(Type type, VisionarayGlobalState *state);
  ~VisionaraySceneImpl();
  void commit();
  void release();
  bool isValid() const;

  void attachInstance(dco::Instance inst, unsigned instID, unsigned userID=~0u);
  void attachGeometry(dco::Geometry geom, unsigned geomID, unsigned userID=~0u);
  void attachGeometry(
      dco::Geometry geom, dco::Material mat, unsigned geomID, unsigned userID=~0u);
  void attachVolume(dco::Volume vol, unsigned geomID, unsigned userID=~0u);
  void updateGeometry(dco::Geometry geom);
  void updateVolume(dco::Volume vol);
  void attachLight(dco::Light light, unsigned id);
  aabb getBounds() const;
  index_bvh_ref_t<dco::BLS> refBVH();

 private:
  void dispatch();

  VisionarayGlobalState *deviceState();
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state);

} // namespace visionaray
