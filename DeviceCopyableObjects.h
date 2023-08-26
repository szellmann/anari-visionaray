
#pragma once

// visionaray
#include "visionaray/texture/texture.h"
#include "visionaray/aligned_vector.h"
#include "visionaray/bvh.h"
#include "visionaray/material.h"
#include "visionaray/matrix_camera.h"
#include "visionaray/pinhole_camera.h"
// ours
#include "common.h"

namespace visionaray {

struct VisionaraySceneImpl;
typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;

} // namespace visionaray

namespace visionaray::dco {

// BLS primitive //

struct BLS
{
  enum Type { Triangle, Sphere, Cylinder, Instance, };
  Type type;
  index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
  index_bvh<basic_sphere<float>>::bvh_ref asSphere;
  index_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
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
    mat3f rot = inverse(bls.asInstance.affine_inv());
    vec3f trans = -bls.asInstance.trans_inv();
    auto verts = compute_vertices(bound);
    aabb result;
    result.invalidate();
    for (vec3 v : verts) {
      v = rot * v + trans;
      result.insert(v);
    }
    return result;
  }

  aabb inval;
  inval.invalidate();
  return inval;
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &ray, const BLS &bls)
{
  if (bls.type == BLS::Triangle)
    return intersect(ray,bls.asTriangle);
  if (bls.type == BLS::Sphere)
    return intersect(ray,bls.asSphere);
  if (bls.type == BLS::Cylinder)
    return intersect(ray,bls.asCylinder);
  else if (bls.type == BLS::Instance) {
    return intersect(ray,bls.asInstance);
  }

  return {};
}

// TLS //

typedef index_bvh<BLS>::bvh_ref TLS;

// Geometry //

struct Geometry
{
  enum Type { Triangle, Sphere, Cylinder, Instance, };
  Type type;
  struct {
    basic_triangle<3,float> *data{nullptr};
    size_t len{0};
  } asTriangle;
  struct {
    unsigned instID{UINT_MAX};
    unsigned groupID{UINT_MAX};
    VisionarayScene scene{nullptr};
    mat4 xfm;
  } asInstance;
};

// Group //

struct Group
{
  unsigned groupID{UINT_MAX};
  Geometry *geoms{nullptr};
};

// Instance //

struct Instance
{
  unsigned instID{UINT_MAX};
  unsigned groupID{UINT_MAX};
  mat4 xfm;
};

// Material //

struct Material
{
  enum Type { Matte, };
  Type type;
  unsigned matID{UINT_MAX};
  struct {
    matte<float> data;
    texture_ref<unorm<8>, 2> colorSampler;
  } asMatte;
};

// Camera //

struct Camera
{
  enum Type { Matrix, Pinhole, };
  Type type;
  unsigned camID{UINT_MAX};
  matrix_camera asMatrixCam;
  pinhole_camera asPinholeCam;
};

} // namespace visionaray::dco
