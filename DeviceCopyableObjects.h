
#pragma once

// visionaray
#include "visionaray/texture/texture.h"
#include "visionaray/bvh.h"
#include "visionaray/directional_light.h"
#include "visionaray/material.h"
#include "visionaray/matrix_camera.h"
#include "visionaray/point_light.h"
#include "visionaray/spot_light.h"
#include "visionaray/thin_lens_camera.h"
// ours
#include "scene/volume/spatial_field/Plane.h"
#include "scene/volume/spatial_field/UElems.h"
#include "common.h"
#include "sampleCDF.h"

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
    Quad = 0x2,
    Sphere = 0x4,
    Cylinder = 0x8,
    Volume = 0x10,
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
  result.t = max(ray.tmin,hr.tnear);
  result.geom_id = vol.volID;
  return result;
}

// BLS primitive //

struct BLS
{
  enum Type { Triangle, Quad, Sphere, Cylinder, Volume, Instance, };
  Type type;
  index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
  index_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
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
  if (bls.type == BLS::Quad && bls.asQuad.num_nodes())
    return bls.asQuad.node(0).get_bounds();
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
  if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    return intersect(ray,bls.asQuad);
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
  ray.intersectionMask = Ray::Triangle | Ray::Quad | Ray::Sphere | Ray::Cylinder;
  return intersect(ray, tls);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectVolumes(
    Ray ray, const TLS &tls)
{
  ray.intersectionMask = Ray::Volume;
  return intersect(ray, tls);
}

// Array //

struct Array
{
  void *data{nullptr};
  size_t len{0};
  ANARIDataType type{ANARI_UNKNOWN};
};

enum class Attribute
{
  _0, _1, _2, _3, Color, None,
};

// Surface //

struct Surface
{
  unsigned surfID{UINT_MAX};
  unsigned geomID{UINT_MAX};
  unsigned matID{UINT_MAX};
};

// Geometry //

struct Geometry
{
  enum Type { Triangle, Quad, Sphere, Cylinder, Volume, Instance, };
  Type type;
  unsigned geomID{UINT_MAX};
  struct {
    basic_triangle<3,float> *data{nullptr};
    size_t len{0};
    Array vertexAttributes[5];
    Array index;
  } asTriangle;
  struct {
    basic_triangle<3,float> *data{nullptr};
    size_t len{0};
  } asQuad;
  struct {
    basic_sphere<float> *data{nullptr};
    size_t len{0};
    Array vertexAttributes[5];
    Array index;
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

  Array primitiveAttributes[5];
};

// Instance //

struct Instance
{
  unsigned instID{UINT_MAX};
  unsigned groupID{UINT_MAX};
  mat4 xfm;
  mat4 invXfm;
};

// Sampler //

struct Sampler
{
  enum Type { Image1D, Image2D, };
  Type type;
  unsigned samplerID{UINT_MAX};
  Attribute inAttribute{Attribute::_0};
  mat4 inTransform{mat4::identity()};
  float4 inOffset{0.f};
  mat4 outTransform{mat4::identity()};
  float4 outOffset{0.f};
  texture_ref<vector<4, unorm<8>>, 1> asImage1D;
  texture_ref<vector<4, unorm<8>>, 2> asImage2D;

  VSNRAY_FUNC
  bool isValid() const
  {
    return samplerID < UINT_MAX &&
        (type == Image1D && asImage1D) ||
        (type == Image2D && asImage2D);
  }
};

// Material //

struct Material
{
  enum Type { Matte, };
  Type type;
  unsigned matID{UINT_MAX};
  Attribute colorAttribute{Attribute::None};
  struct {
    unsigned samplerID{UINT_MAX};
    matte<float> data;
  } asMatte;
};

VSNRAY_FUNC
inline Material makeDefaultMaterial()
{
  Material mat;
  mat.type = Material::Matte;
  mat.asMatte.data.cd() = from_rgb(vec3(0,1,0));
  mat.asMatte.data.kd() = 1.f;
  return mat;
}

// Group //

struct Group
{
  unsigned groupID{UINT_MAX};
  Geometry *geoms{nullptr};
  Material *materials{nullptr};
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
      return objectToLocal(object + float3(0.5f) * spacing) / float3(dims);
    }

  } asStructuredRegular;
  struct {
    index_bvh<UElem>::bvh_ref samplingBVH;
  } asUnstructured;
};

// Grid accelerator to traverse spatial fields //

struct GridAccel
{
  unsigned fieldID{UINT_MAX}; // the field this grid belongs to
  int3 dims;
  box3 worldBounds;
  box1 *valueRanges; // min/max ranges
  float *maxOpacities; // used as majorants
};

// Transfer functions //

struct TransferFunction
{
  enum Type { _1D, };
  Type type;
  unsigned volID{UINT_MAX};
  struct {
    unsigned numValues;
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
  thin_lens_camera asPinholeCam;
};

// Light //

struct Light
{
  enum Type { Directional, Point, Spot, HDRI, };
  Type type;
  unsigned lightID{UINT_MAX};
  bool visible{true};
  directional_light<float> asDirectional;
  point_light<float> asPoint;
  spot_light<float> asSpot;
  struct {
    texture_ref<float3, 2> radiance;
    float scale{1.f};
    struct CDF {
      float *rows{nullptr};
      float *lastCol{nullptr};
      unsigned width{0};
      unsigned height{0};
    } cdf;

    template <typename RNG>
    VSNRAY_FUNC
    inline light_sample<float> sample(const float3 &refPoint, RNG &rng) const
    {
      CDFSample sample = sampleCDF(cdf.rows, cdf.lastCol, cdf.width, cdf.height, rng(), rng());
      float invjacobian = cdf.width*cdf.height/float(4*M_PI);
      float3 L(toPolar(float2(sample.x/float(cdf.width), sample.y/float(cdf.height))));
      light_sample<float> ls;
      ls.dir = L;
      ls.dist = FLT_MAX;
      ls.pdf = sample.pdfx*sample.pdfy*invjacobian;
      return ls;
    }

    VSNRAY_FUNC
    inline float3 intensity(const float3 dir)
    {
      return tex2D(radiance, toUV(dir))*scale;
    }

  } asHDRI;
};

} // namespace visionaray::dco
