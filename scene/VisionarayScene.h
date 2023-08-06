
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

struct BLS
{
  enum Type { Triangle, Sphere, Cylinder, Instance, };
  Type type;
  TriangleBVH::bvh_ref asTriangle;
  SphereBVH::bvh_ref asSphere;
  CylinderBVH::bvh_ref asCylinder;
  index_bvh<BLS>::bvh_inst asInstance;
};

VSNRAY_FUNC
inline aabb get_bounds(const BLS &bls)
{
  if (bls.type == BLS::Triangle && bls.asTriangle.num_nodes())
    return bls.asTriangle.node(0).get_bounds();
  else if (bls.type == BLS::Sphere && bls.asSphere.num_nodes())
    return bls.asSphere.node(0).get_bounds();
  else if (bls.type == BLS::Cylinder && bls.asCylinder.num_nodes())
    return bls.asCylinder.node(0).get_bounds();
  else if (bls.type == BLS::Instance && bls.asInstance.num_nodes()) {
    aabb bound = bls.asInstance.node(0).get_bounds();
    std::cout << bound.min << ',' << bound.max << '\n';
    mat3f rot = inverse(bls.asInstance.affine_inv());
    vec3f trans = -bls.asInstance.trans_inv();
    bound.min = rot * (bound.min + trans);
    bound.max = rot * (bound.max + trans);
    return bound;
  }

  aabb inval;
  inval.invalidate();
  return inval;
}

typedef index_bvh<BLS> TLS;

struct VisionaraySceneImpl
{
  // Geometries //
  aligned_vector<VisionarayGeometry> m_geometries;

  // Accels //
  TLS m_TLS;
  aligned_vector<BLS> m_BLSs;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH> triangleBLSs;
    aligned_vector<SphereBVH>   sphereBLSs;
    aligned_vector<CylinderBVH> cylinderBLSs;
  } m_accelStorage;

  // Surface properties //
  aligned_vector<VisionarayMaterial> m_materials;

  void commit();
  void release();
  void attachGeometry(VisionarayGeometry geom, unsigned geomID);
};

typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;
VisionarayScene newVisionarayScene();

} // namespace visionaray
