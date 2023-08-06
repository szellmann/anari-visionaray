
#pragma once

// std
#include <memory>
// visionaray
#include "visionaray/aligned_vector.h"
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"

namespace visionaray {

typedef index_bvh<basic_triangle<3,float>> TriangleBVH;
typedef index_bvh<basic_sphere<float>>     SphereBVH;
typedef index_bvh<basic_cylinder<float>>   CylinderBVH;

typedef index_bvh<typename TriangleBVH::bvh_inst> TriangleTLS;
typedef index_bvh<typename SphereBVH::bvh_inst>   SphereTLS;
typedef index_bvh<typename CylinderBVH::bvh_inst> CylinderTLS;

struct VisionaraySceneImpl
{
  // Geometries //
  aligned_vector<VisionarayGeometry> m_geometries;

  struct {
    TriangleTLS::bvh_ref triangleTLS;
    SphereTLS::bvh_ref   sphereTLS;
    CylinderTLS::bvh_ref cylinderTLS;
  } m_TLSs;

  // Surface properties //
  aligned_vector<VisionarayMaterial> m_materials;

  void commit();
  void release();
  void attachGeometry(VisionarayGeometry geom, unsigned geomID);
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene();

} // namespace visionaray
