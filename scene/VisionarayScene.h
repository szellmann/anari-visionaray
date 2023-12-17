
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

namespace visionaray {

struct VisionaraySceneImpl
{
  typedef index_bvh<basic_triangle<3,float>> TriangleBVH;
  typedef index_bvh<basic_triangle<3,float>> QuadBVH;
  typedef index_bvh<basic_sphere<float>>     SphereBVH;
  typedef index_bvh<basic_cylinder<float>>   CylinderBVH;
  typedef index_bvh<dco::ISOSurface>         ISOSurfaceBVH;
  typedef index_bvh<dco::Volume>             VolumeBVH;

  typedef index_bvh<dco::BLS> TLS;

  enum Type { World, Group, };
  Type type;

  // Surface data //
  DeviceObjectArray<dco::Geometry> m_geometries;
  DeviceObjectArray<dco::Material> m_materials;
  DeviceObjectArray<dco::Light>    m_lights;

  // Accels //
  TLS m_TLS;
  DeviceObjectArray<dco::BLS> m_BLSs;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH>   triangleBLSs;
    aligned_vector<QuadBVH>       quadBLSs;
    aligned_vector<SphereBVH>     sphereBLSs;
    aligned_vector<CylinderBVH>   cylinderBLSs;
    aligned_vector<ISOSurfaceBVH> isoSurfaceBLSs;
    aligned_vector<VolumeBVH>     volumeBLSs;
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
  void attachGeometry(dco::Geometry geom, unsigned geomID);
  void attachGeometry(dco::Geometry geom, dco::Material mat, unsigned geomID);
  void updateGeometry(dco::Geometry geom);
  void attachLight(dco::Light light, unsigned lightID);

 private:
  void dispatch();

  VisionarayGlobalState *deviceState();
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene(
    VisionaraySceneImpl::Type type, VisionarayGlobalState *state);

} // namespace visionaray
