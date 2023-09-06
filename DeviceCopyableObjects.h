
#pragma once

// visionaray
#include "visionaray/texture/texture.h"
#include "visionaray/aligned_vector.h"
#include "visionaray/bvh.h"
#include "visionaray/material.h"
#include "visionaray/matrix_camera.h"
#include "visionaray/pinhole_camera.h"
#include "visionaray/point_light.h"
// ours
#include "scene/volume/spatial_field/Plane.h"
#include "scene/volume/spatial_field/UElems.h"
#include "common.h"

namespace visionaray {

// Forward decls //

struct VisionaraySceneImpl;
typedef std::shared_ptr<VisionaraySceneImpl> VisionarayScene;

// Ray //

struct Ray : basic_ray<float>
{
  enum IntersectionMask {
    All = 0xffffffff,
    Triangle = 0x1,
    Sphere = 0x2,
    Cylinder = 0x4,
    Volume = 0x8,
  };
  unsigned intersectionMask = All;
};

} // namespace visionaray

namespace visionaray::dco {

// Volume //

struct Volume
{
  enum Type { TransferFunction1D, };
  Type type;

  enum FieldType { Unstructured, };
  FieldType fieldType;

  unsigned volID{UINT_MAX};
  unsigned fieldID{UINT_MAX}; // _should_ be same as volID

  aabb bounds;
};

VSNRAY_FUNC
inline aabb get_bounds(const Volume &vol)
{
  return vol.bounds;
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const Volume &vol)
{
  assert(0);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &ray, const Volume &vol)
{
  auto hr = intersect(ray,vol.bounds);
  // we just report that we did hit the box; the user
  // is later expected to intersect the volume bounds
  // themselves to compute [t0,t1]
  hit_record<Ray, primitive<unsigned>> result;
  result.hit = hr.hit;
  result.t = hr.tnear;
  result.geom_id = vol.volID;
  return result;
}

// BLS primitive //

struct BLS
{
  enum Type { Triangle, Sphere, Cylinder, Volume, Instance, };
  Type type;
  index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
  index_bvh<basic_sphere<float>>::bvh_ref asSphere;
  index_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
  index_bvh<dco::Volume>::bvh_ref asVolume;
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
  else if (bls.type == BLS::Volume && bls.asVolume.num_nodes())
    return bls.asVolume.node(0).get_bounds();
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
  if (bls.type == BLS::Triangle && (ray.intersectionMask & Ray::Triangle))
    return intersect(ray,bls.asTriangle);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    return intersect(ray,bls.asSphere);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    return intersect(ray,bls.asCylinder);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::Volume))
    return intersect(ray,bls.asVolume);
  else if (bls.type == BLS::Instance)
    return intersect(ray,bls.asInstance);

  return {};
}

// TLS //

typedef index_bvh<BLS>::bvh_ref TLS;

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    Ray ray, const TLS &tls)
{
  ray.intersectionMask = Ray::Triangle | Ray::Sphere | Ray::Cylinder;
  return intersect(ray, tls);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectVolumes(
    Ray ray, const TLS &tls)
{
  ray.intersectionMask = Ray::Volume;
  return intersect(ray, tls);
}

// Geometry //

struct Geometry
{
  enum Type { Triangle, Sphere, Cylinder, Volume, Instance, };
  Type type;
  struct {
    basic_triangle<3,float> *data{nullptr};
    size_t len{0};
  } asTriangle;
  struct {
    basic_sphere<float> *data{nullptr};
    size_t len{0};
  } asSphere;
  struct {
    basic_cylinder<float> *data{nullptr};
    size_t len{0};
  } asCylinder;
  struct {
    dco::Volume data;
  } asVolume;
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

// Unstructured element primitive //

struct UElem
{
  uint64_t begin;
  uint64_t end;
  uint64_t elemID;
  uint64_t *indexBuffer;
  float4 *vertexBuffer;
};

VSNRAY_FUNC
inline aabb get_bounds(const UElem &elem)
{
  aabb result;
  result.invalidate();
  for (uint64_t i=elem.begin;i<elem.end;++i) {
    result.insert(elem.vertexBuffer[elem.indexBuffer[i]].xyz());
  }
  return result;
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const UElem &elem)
{
  assert(0);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &ray, const UElem &elem)
{
  uint64_t numVerts = elem.end-elem.begin;

  float4 v[8];
  for (int i=0; i<numVerts; ++i) {
    uint64_t idx = elem.indexBuffer[elem.begin+i];
    v[i] = elem.vertexBuffer[idx];
  }

  float3 pos = ray.ori;
  float value;
  bool hit=numVerts==4 && intersectTet(value,pos,v[0],v[1],v[2],v[3])
        || numVerts==5 && intersectPyrEXT(value,pos,v[0],v[1],v[2],v[3],v[4])
        || numVerts==6 && intersectWedgeEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5])
        || numVerts==8 && intersectHexEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);

  hit_record<Ray, primitive<unsigned>> result;
  result.hit = hit;
  if (hit) {
    result.t = 0.f;
    result.prim_id = elem.elemID;
    result.u = value; // misuse "u" to store value
  }
  return result;
}
// Spatial Field //

struct SpatialField
{
  enum Type { StructuredRegular, Unstructured, };
  Type type;
  unsigned fieldID{UINT_MAX};
  float baseDT{0.5f};
  struct {
    texture_ref<float, 3> sampler;
    float3 origin{0.f,0.f,0.f}, spacing{1.f,1.f,1.f};
    uint3 dims{0,0,0};

    VSNRAY_FUNC
    inline float3 objectToLocal(const float3 &object) const
    {
      return 1.f / (spacing) * (object - origin);
    }

    VSNRAY_FUNC
    inline float3 objectToTexCoord(const float3 &object) const
    {
      return objectToLocal(object) / float3(dims);
    }

  } asStructuredRegular;
  struct {
    index_bvh<UElem>::bvh_ref samplingBVH;
  } asUnstructured;
};

// Transfer functions //

struct TransferFunction
{
  enum Type { _1D, };
  Type type;
  unsigned volID{UINT_MAX};
  struct {
    box1 valueRange{0.f, 1.f};
    texture_ref<float4, 1> sampler;
  } as1D;
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

// Light //

struct Light
{
  enum Type { Point, };
  Type type;
  unsigned lightID{UINT_MAX};
  point_light<float> asPoint;
};

} // namespace visionaray::dco
