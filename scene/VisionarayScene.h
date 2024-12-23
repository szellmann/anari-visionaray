
#pragma once

// std
#include <memory>
// visionaray
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"
#include "light/Light.h"
#include "DeviceCopyableObjects.h"
#if defined(WITH_CUDA) || defined(WITH_HIP)
#include "VisionaraySceneGPU.h"
#endif

namespace visionaray {

struct VisionaraySceneImpl
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  friend struct VisionaraySceneGPU;
#endif

#if defined(WITH_CUDA) || defined(WITH_HIP)
  typedef index_bvh<basic_triangle<3,float>>  TriangleBVH;
  typedef index_bvh<basic_triangle<3,float>>  QuadBVH;
  typedef index_bvh<basic_sphere<float>>      SphereBVH;
  typedef index_bvh<dco::Cone>                ConeBVH;
  typedef index_bvh<basic_cylinder<float>>    CylinderBVH;
  typedef index_bvh<dco::BezierCurve>         BezierCurveBVH;
  typedef index_bvh<dco::ISOSurface>          ISOSurfaceBVH;
  typedef index_bvh<dco::Volume>              VolumeBVH;
#else
  typedef index_bvh4<basic_triangle<3,float>> TriangleBVH;
  typedef index_bvh4<basic_triangle<3,float>> QuadBVH;
  typedef index_bvh4<basic_sphere<float>>     SphereBVH;
  typedef index_bvh4<dco::Cone>               ConeBVH;
  typedef index_bvh4<basic_cylinder<float>>   CylinderBVH;
  typedef index_bvh4<dco::BezierCurve>        BezierCurveBVH;
  typedef index_bvh4<dco::ISOSurface>         ISOSurfaceBVH;
  typedef index_bvh4<dco::Volume>             VolumeBVH;
#endif

  typedef index_bvh<dco::BLS> TLS;
  typedef index_bvh<dco::Instance> WorldTLS;

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
  DeviceHandleArray m_allLights;

  // Accels //
  TLS m_TLS;
  WorldTLS m_worldTLS;
  DeviceObjectArray<dco::BLS> m_BLSs;
  DeviceObjectArray<dco::Instance> m_worldBLSs;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH>    triangleBLSs;
    aligned_vector<QuadBVH>        quadBLSs;
    aligned_vector<SphereBVH>      sphereBLSs;
    aligned_vector<ConeBVH>        coneBLSs;
    aligned_vector<CylinderBVH>    cylinderBLSs;
    aligned_vector<BezierCurveBVH> bezierCurveBLSs;
    aligned_vector<ISOSurfaceBVH>  isoSurfaceBLSs;
    aligned_vector<VolumeBVH>      volumeBLSs;
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
#ifdef WITH_CUDA
  cuda_index_bvh<dco::BLS>::bvh_ref refBVH();
#elif defined(WITH_HIP)
  hip_index_bvh<dco::BLS>::bvh_ref refBVH();
#else
  index_bvh<dco::BLS>::bvh_ref refBVH();
#endif

 private:
  void dispatch();

  VisionarayGlobalState *deviceState();
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::unique_ptr<VisionaraySceneGPU> m_gpuScene{nullptr};
#endif
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state);

} // namespace visionaray
