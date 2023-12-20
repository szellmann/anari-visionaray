
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
#include "frame/common.h"
#include "renderer/DDA.h"
#include "scene/volume/spatial_field/Plane.h"
#include "scene/volume/spatial_field/UElems.h"
#include "scene/volume/spatial_field/UElemGrid.h"
#include "common.h"
#include "sampleCDF.h"

namespace visionaray {

namespace dco {
typedef uint32_t Handle;
VSNRAY_FUNC
inline bool validHandle(Handle hnd)
{ return hnd < UINT_MAX; }
} // namespace dco
typedef dco::Handle DeviceObjectHandle;

// Ray //

struct Ray : basic_ray<float>
{
  enum IntersectionMask {
    All = 0xffffffff,
    Triangle = 0x1,
    Quad = 0x2,
    Sphere = 0x4,
    Cylinder = 0x8,
    ISOSurface = 0x10,
    Volume = 0x20,
  };
  unsigned intersectionMask = All;

#if 1
  bool dbg{false};
  VSNRAY_FUNC inline bool debug() const {
    return dbg;
  }
#endif
};

} // namespace visionaray

namespace visionaray::dco {

// Unstructured element primitive //

struct UElem
{
  uint64_t begin;
  uint64_t end;
  uint64_t elemID;
  const uint64_t *indexBuffer;
  float4 *vertexBuffer;
  // "stitcher" extension
  int3 *gridDimsBuffer;
  aabb *gridDomainsBuffer;
  uint64_t *gridScalarsOffsetBuffer;
  float *gridScalarsBuffer;
};

VSNRAY_FUNC
inline aabb get_bounds(const UElem &elem)
{
  aabb result;
  if (elem.end-elem.begin > 0) {
    result.invalidate();
    for (uint64_t i=elem.begin;i<elem.end;++i) {
      result.insert(elem.vertexBuffer[elem.indexBuffer[i]].xyz());
    }
  } else { // no vertices -> voxel grid
    result = elem.gridDomainsBuffer[elem.elemID];
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
  hit_record<Ray, primitive<unsigned>> result;
  float3 pos = ray.ori;
  float value = 0.f;

  uint64_t numVerts = elem.end-elem.begin;

  if (numVerts > 0) { // regular uelem
    float4 v[8];
    for (int i=0; i<numVerts; ++i) {
      uint64_t idx = elem.indexBuffer[elem.begin+i];
      v[i] = elem.vertexBuffer[idx];
    }

    bool hit=numVerts==4 && intersectTet(value,pos,v[0],v[1],v[2],v[3])
          || numVerts==5 && intersectPyrEXT(value,pos,v[0],v[1],v[2],v[3],v[4])
          || numVerts==6 && intersectWedgeEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5])
          || numVerts==8 && intersectHexEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);

    result.hit = hit;
  } else {
    // element is a voxel grid (for "stitcher" AMR data)
    int3 dims = elem.gridDimsBuffer[elem.elemID];
    aabb domain = elem.gridDomainsBuffer[elem.elemID];
    uint64_t scalarsOffset = elem.gridScalarsOffsetBuffer[elem.elemID];

    bool hit = intersectGrid(dims, domain, scalarsOffset, elem.gridScalarsBuffer,
                             pos, value);
    result.hit = hit;
  }

  if (result.hit) {
    result.t = 0.f;
    result.prim_id = elem.elemID;
    result.u = value; // misuse "u" to store value
  }

  return result;
}

// Grid accelerator to traverse spatial fields //

struct GridAccel
{
  unsigned fieldID{UINT_MAX}; // the field this grid belongs to
  int3 dims;
  box3 worldBounds;
  box1 *valueRanges; // min/max ranges
  float *maxOpacities; // used as majorants
};

// Spatial Field //

struct SpatialField
{
  enum Type { StructuredRegular, Unstructured, Unknown, };
  Type type{Unknown};
  unsigned fieldID{UINT_MAX};
  float baseDT{0.5f};
  GridAccel gridAccel;
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

VSNRAY_FUNC
inline bool sampleField(SpatialField sf, vec3 P, float &value) {
  if (sf.type == SpatialField::StructuredRegular) {
    value = tex3D(sf.asStructuredRegular.sampler,
        sf.asStructuredRegular.objectToTexCoord(P));
    return true;
  } else if (sf.type == SpatialField::Unstructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;
    auto hr = intersect(ray, sf.asUnstructured.samplingBVH);

    if (!hr.hit)
      return false;

    value = hr.u; // value is stored in "u"!
    return true;
  }

  return false;
}

VSNRAY_FUNC
inline bool sampleGradient(SpatialField sf, vec3 P, float3 &value) {
  float x0=0, x1=0, y0=0, y1=0, z0=0, z1=0;
  bool b0 = sampleField(sf, P+float3{sf.baseDT, 0.f, 0.f}, x1);
  bool b1 = sampleField(sf, P-float3{sf.baseDT, 0.f, 0.f}, x0);
  bool b2 = sampleField(sf, P+float3{0.f, sf.baseDT, 0.f}, y1);
  bool b3 = sampleField(sf, P-float3{0.f, sf.baseDT, 0.f}, y0);
  bool b4 = sampleField(sf, P+float3{0.f, 0.f, sf.baseDT}, z1);
  bool b5 = sampleField(sf, P-float3{0.f, 0.f, sf.baseDT}, z0);
  if (b0 && b1 && b2 && b3 && b4 && b5) {
    value = float3{x1,y1,z1}-float3{x0,y0,z0};
    return true; // TODO
  } else {
    value = float3{0.f};
    return false;
  }
}

// Volume //

struct Volume
{
  enum Type { TransferFunction1D, Unknown, };
  Type type{Unknown};

  unsigned volID{UINT_MAX};
  unsigned geomID{UINT_MAX}; // ID in group (internally realized as geom)

  struct {
    unsigned fieldID{UINT_MAX}; // _should_ be same as volID
  } asTransferFunction1D;

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
  result.geom_id = vol.geomID;
  return result;
}

// ISO surface //

struct ISOSurface
{
  unsigned isoID{UINT_MAX};
  unsigned geomID{UINT_MAX};

  SpatialField field;
  unsigned numValues{0};
  const float *values{nullptr};

  aabb bounds;
};

VSNRAY_FUNC
inline aabb get_bounds(const ISOSurface &iso)
{
  return iso.bounds;
}

inline void split_primitive(
    aabb &L, aabb &R, float plane, int axis, const ISOSurface &vol)
{
  assert(0);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    Ray ray, const ISOSurface &iso)
{
  hit_record<Ray, primitive<unsigned>> result;

  auto boxHit = intersect(ray, iso.bounds);
  if (!boxHit.hit)
    return result;

  float dt = iso.field.baseDT;

  auto isectFunc = [&](const int leafID, float t0, float t1) {
    bool empty = (leafID != -1);

    if (leafID >= 0 && iso.field.gridAccel.valueRanges) {
      box1 valueRange = iso.field.gridAccel.valueRanges[leafID];
      for (unsigned i=0;i<iso.numValues;i++) {
        float isoValue = iso.values[i];
        if (valueRange.min <= isoValue && isoValue < valueRange.max) {
          empty = false;
          break;
        }
      }
    }

    if (empty)
      return true;

    float t0_old = t0;
    float t1_old = t1;
    t0 = t1 = boxHit.tnear-dt/2.f;
    while (t0 < t0_old) t0 += dt;
    while (t1 < t1_old) t1 += dt;

    for (float t=t0;t<t1;t+=dt) {
      float3 P1 = ray.ori+ray.dir*t;
      float3 P2 = ray.ori+ray.dir*(t+dt);
      float v1 = 0.f, v2 = 0.f;
      if (sampleField(iso.field,P1,v1)
       && sampleField(iso.field,P2,v2)) {
        unsigned numISOs = iso.numValues;
        bool hit=false;
        for (unsigned i=0;i<numISOs;i++) {
          float isoValue = iso.values[i];
          if ((v1 <= isoValue && v2 > isoValue) || (v2 <= isoValue && v1 > isoValue)) {
            float tHit = t+dt/2.f;
            if (tHit < result.t) {
              result.hit = true;
              result.prim_id = i;
              result.geom_id = iso.geomID;
              result.t = tHit;
            }
            hit = true;
          }
        }
        if (hit) return false; // stop traversal
      }
    }

    return true; // cont. traversal to the next spat. partition
  };

  ray.tmin = boxHit.tnear;
  ray.tmax = boxHit.tfar;
  if (iso.field.type == dco::SpatialField::Unstructured ||
      iso.field.type == dco::SpatialField::StructuredRegular)
    dda3(ray, iso.field.gridAccel.dims, iso.field.gridAccel.worldBounds, isectFunc);
  else
    isectFunc(-1, boxHit.tnear, boxHit.tfar);

  return result;
}

// BLS primitive //

struct BLS
{
  enum Type { Triangle, Quad, Sphere, Cylinder, ISOSurface, Volume, Instance, Unknown, };
  Type type{Unknown};
  unsigned blsID{UINT_MAX};
  index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
  index_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
  index_bvh<basic_sphere<float>>::bvh_ref asSphere;
  index_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
  index_bvh<dco::ISOSurface>::bvh_ref asISOSurface;
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
  else if (bls.type == BLS::ISOSurface && bls.asISOSurface.num_nodes())
    return bls.asISOSurface.node(0).get_bounds();
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
  else if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    return intersect(ray,bls.asQuad);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    return intersect(ray,bls.asSphere);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    return intersect(ray,bls.asCylinder);
  else if (bls.type == BLS::ISOSurface && (ray.intersectionMask & Ray::ISOSurface))
    return intersect(ray,bls.asISOSurface);
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
  ray.intersectionMask
      = Ray::Triangle | Ray::Quad | Ray::Sphere | Ray::Cylinder | Ray::ISOSurface;
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
  const void *data{nullptr};
  size_t len{0};
  ANARIDataType type{ANARI_UNKNOWN};
};

enum class Attribute
{
  _0, _1, _2, _3, Color, None,
};

// Instance //

struct Instance
{
  unsigned instID{UINT_MAX};
  unsigned groupID{UINT_MAX};
  index_bvh<BLS>::bvh_inst instBVH;
  mat4 xfm;
  mat4 invXfm;
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
  enum Type { Triangle, Quad, Sphere, Cylinder, ISOSurface, Volume, Instance, Unknown, };
  Type type{Unknown};
  unsigned geomID{UINT_MAX};
  bool updated{false};
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
    Array vertexAttributes[5];
    Array index;
  } asCylinder;
  struct {
    dco::ISOSurface data;
  } asISOSurface;
  struct {
    dco::Volume data;
  } asVolume;
  struct {
    dco::Instance data;
  } asInstance;

  Array primitiveAttributes[5];

  VSNRAY_FUNC
  inline bool isValid() const
  {
    if (type == ISOSurface) {
      return asISOSurface.data.numValues > 0;
    }
    // TODO..
    return true;
  }

  VSNRAY_FUNC
  inline void setUpdated(const bool up)
  {
    updated = up;
  }
};

// Sampler //

struct Sampler
{
  enum Type { Image1D, Image2D, Unknown, };
  Type type{Unknown};
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
  enum Type { Matte, Unknown, };
  Type type{Unknown};
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

// Light //

struct Light
{
  enum Type { Directional, Point, Spot, HDRI, Unknown, };
  Type type{Unknown};
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
    inline float3 intensity(const float3 dir) const
    {
      return tex2D(radiance, toUV(dir))*scale;
    }

  } asHDRI;
};

// Group //

struct Group
{
  unsigned groupID{UINT_MAX};

  unsigned numBLSs{0};
  dco::BLS *BLSs{nullptr};
  unsigned numGeoms{0};
  Handle *geoms{nullptr};
  unsigned numMaterials{0};
  Handle *materials{nullptr};
  unsigned numLights{0};
  Handle *lights{nullptr};
};

// Transfer functions //

struct TransferFunction
{
  enum Type { _1D, Unknown, };
  Type type{Unknown};

  unsigned tfID{UINT_MAX};
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
  enum Type { Matrix, Pinhole, Unknown, };
  Type type{Unknown};
  unsigned camID{UINT_MAX};
  matrix_camera asMatrixCam;
  thin_lens_camera asPinholeCam;
};

// Frame //

struct Frame
{
  unsigned frameID{UINT_MAX};
  unsigned frameCounter{0};
  uint2 size;
  float2 invSize;
  int perPixelBytes{1};
  bool stochasticRendering{false};

  anari::DataType colorType{ANARI_UNKNOWN};
  anari::DataType depthType{ANARI_UNKNOWN};
  anari::DataType normalType{ANARI_UNKNOWN};
  anari::DataType albedoType{ANARI_UNKNOWN};
  anari::DataType primIdType{ANARI_UNKNOWN};
  anari::DataType objIdType{ANARI_UNKNOWN};
  anari::DataType instIdType{ANARI_UNKNOWN};

  uint8_t *pixelBuffer{nullptr};
  float *depthBuffer{nullptr};
  float3 *normalBuffer{nullptr};
  float3 *albedoBuffer{nullptr};
  float4 *motionVecBuffer{nullptr};
  uint32_t *primIdBuffer{nullptr};
  uint32_t *objIdBuffer{nullptr};
  uint32_t *instIdBuffer{nullptr};
  float4 *accumBuffer{nullptr};

  struct {
    bool enabled{false};
    float alpha{0.3f};
    float4 *currBuffer{nullptr};
    float4 *prevBuffer{nullptr};
    float3 *currAlbedoBuffer{nullptr};
    float3 *prevAlbedoBuffer{nullptr};
    texture_ref<float4, 2> history;
  } taa;

  VSNRAY_FUNC
  inline PixelSample pixelSample(int x, int y)
  {
    const auto idx = y * size.x + x;

    PixelSample s;

    if (taa.enabled) {
      if (taa.currBuffer)
        s.color = taa.currBuffer[idx];
      if (taa.currAlbedoBuffer)
        s.albedo = taa.currAlbedoBuffer[idx];
    } else {
      if (accumBuffer)
        s.color = accumBuffer[idx];
      if (albedoBuffer)
        s.albedo = albedoBuffer[idx];
    }

    if (depthBuffer)
      s.depth = depthBuffer[idx];
    if (normalBuffer)
      s.Ng = normalBuffer[idx];
    if (motionVecBuffer)
      s.motionVec = motionVecBuffer[idx];
    if (primIdBuffer)
      s.primId = primIdBuffer[idx];
    if (objIdBuffer)
      s.objId = objIdBuffer[idx];
    if (instIdBuffer)
      s.instId = instIdBuffer[idx];

    return s;
  }

  VSNRAY_FUNC
  inline PixelSample accumSample(int x, int y, int accumID, PixelSample s)
  {
    const auto idx = y * size.x + x;

    if (taa.enabled) {
      int2 prevID = int2(float2(x,y) + motionVecBuffer[idx].xy());
      prevID = clamp(prevID, int2(0), int2(size)-int2(1));
      const auto prevIdx = prevID.y * size.x + prevID.x;
      float alpha = taa.alpha;
      if (!(fabsf(taa.prevAlbedoBuffer[prevIdx].x-taa.currAlbedoBuffer[idx].x) < 1e-2f
         && fabsf(taa.prevAlbedoBuffer[prevIdx].y-taa.currAlbedoBuffer[idx].y) < 1e-2f
         && fabsf(taa.prevAlbedoBuffer[prevIdx].z-taa.currAlbedoBuffer[idx].z) < 1e-2f)) {
        alpha = 1.f;
      }
      float prevX = x + motionVecBuffer[idx].x;
      float prevY = y + motionVecBuffer[idx].y;
      float2 texCoord((prevX+0.5f)/size.x, (prevY+0.5f)/size.y);
      float4 history = tex2D(taa.history, texCoord);
      taa.currBuffer[idx] = (1-alpha)*history + alpha*s.color;
      s.color = taa.currBuffer[idx];
    } else if (stochasticRendering) {
      float alpha = 1.f / (accumID+1);
      accumBuffer[idx] = (1-alpha)*accumBuffer[idx] + alpha*s.color;
      s.color = accumBuffer[idx];
    }

    return s;
  }

  VSNRAY_FUNC
  inline void toneMap(int x, int y, PixelSample s)
  {
    const auto idx = y * size.x + x;
    auto *color = pixelBuffer + (idx * perPixelBytes);

    switch (colorType) {
    case ANARI_UFIXED8_VEC4: {
      auto c = cvt_uint32(s.color);
      std::memcpy(color, &c, sizeof(c));
      break;
    }
    case ANARI_UFIXED8_RGBA_SRGB: {
      auto c = cvt_uint32_srgb(s.color);
      std::memcpy(color, &c, sizeof(c));
      break;
    }
    case ANARI_FLOAT32_VEC4: {
      std::memcpy(color, &s.color, sizeof(s.color));
      break;
    }
    default:
      break;
    }
  }

  VSNRAY_FUNC
  inline void fillGBuffer(int x, int y, PixelSample s)
  {
    const auto idx = y * size.x + x;

    if (depthBuffer)
      depthBuffer[idx] = s.depth;
    if (normalBuffer)
      normalBuffer[idx] = s.Ng;
    if (albedoBuffer)
      albedoBuffer[idx] = s.albedo;
    if (motionVecBuffer)
      motionVecBuffer[idx] = s.motionVec;
    if (primIdBuffer)
      primIdBuffer[idx] = s.primId;
    if (objIdBuffer)
      objIdBuffer[idx] = s.objId;
    if (instIdBuffer)
      instIdBuffer[idx] = s.instId;
    if (taa.currBuffer)
      taa.currBuffer[idx] = s.color;
    if (taa.currAlbedoBuffer)
      taa.currAlbedoBuffer[idx] = s.albedo;
  }

  VSNRAY_FUNC
  inline void writeSample(int x, int y, int accumID, PixelSample s)
  {
    fillGBuffer(x, y, s);
    toneMap(x, y, accumSample(x, y, accumID, s));
  }
};

} // namespace visionaray::dco
