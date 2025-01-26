
#pragma once

// visionaray
#include "visionaray/bvh.h"
#include "visionaray/directional_light.h"
#include "visionaray/matrix_camera.h"
#include "visionaray/area_light.h"
#include "visionaray/point_light.h"
#include "visionaray/spot_light.h"
#include "visionaray/thin_lens_camera.h"
#if defined(WITH_CUDA)
#include "visionaray/texture/cuda_texture.h"
#elif defined(WITH_HIP)
#include "visionaray/texture/hip_texture.h"
#else
#include "visionaray/texture/texture.h"
#endif
// ours
#include "frame/common.h"
#include "renderer/DDA.h"
#include "scene/volume/spatial_field/Plane.h"
#include "scene/volume/spatial_field/UElems.h"
#include "scene/volume/spatial_field/UElemGrid.h"
#include "common.h"
#include "sampleCDF.h"

#if defined(WITH_CUDA) && !defined(__CUDACC__)
#include <visionaray/cuda/device_vector.h>
namespace visionaray {
// visionaray only defines these when compiling with nvcc:
template <typename P>
using cuda_bvh          = bvh_t<cuda::device_vector<P>, cuda::device_vector<bvh_node>>;
template <typename P>
using cuda_index_bvh    = index_bvh_t<cuda::device_vector<P>, cuda::device_vector<bvh_node>, cuda::device_vector<unsigned>>;
} // namespace visionaray
#endif

#if defined(WITH_HIP) && !defined(__HIPCC__)
#include <visionaray/hip/device_vector.h>
namespace visionaray {
// visionaray only defines these when compiling with hipcc:
template <typename P>
using hip_bvh           = bvh_t<hip::device_vector<P>, hip::device_vector<bvh_node>>;
template <typename P>
using hip_index_bvh     = index_bvh_t<hip::device_vector<P>, hip::device_vector<bvh_node>, hip::device_vector<unsigned>>;
} // namespace visionaray
#endif

#ifdef WITH_NANOVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#endif

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
    Cone = 0x8,
    Cylinder = 0x10,
    Curve = 0x20,
    BezierCurve = 0x40,
    ISOSurface = 0x80,
    Volume = 0x100,
    VolumeBounds = 0x200,
  };
  unsigned intersectionMask = All;
  float time{0.f};
  void *prd{nullptr};

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

// Block primitive //

struct Block
{
  uint32_t ID{UINT_MAX};
  aabbi bounds;
  int level;
  uint32_t scalarOffset;
  box1 valueRange;
  float *scalarsBuffer{nullptr};

  VSNRAY_FUNC
  float getScalar(int ix, int iy, int iz) const
  {
    const int3 blockSize = numCells();
    const uint32_t idx
      = scalarOffset
      + ix
      + iy * blockSize.x
      + iz * blockSize.x*blockSize.y;
    return scalarsBuffer[idx];
  }

  VSNRAY_FUNC
  int cellSize() const
  { return 1<<level; }

  VSNRAY_FUNC
  int3 numCells() const
  { return bounds.max-bounds.min+int3(1); }

  VSNRAY_FUNC
  aabb worldBounds() const
  {
    return aabb(
      float3(bounds.min)*float(cellSize()),
      float3(bounds.max+int3(1))*float(cellSize())
    );
  }

  VSNRAY_FUNC
  aabb filterDomain() const
  {
    const float3 cellSize2(cellSize()*0.5f);
    const aabb wb = worldBounds();
    return aabb(wb.min-cellSize2, wb.max+cellSize2);
  }

  VSNRAY_FUNC
  aabb cellBounds(const vec3i cellID) const
  {
    aabb cb;
    cb.min = float3(bounds.min+cellID)*float(cellSize());
    cb.max = float3(bounds.max+cellID+int3(1))*float(cellSize());
    return cb;
  }
};

VSNRAY_FUNC
inline aabb get_bounds(const Block &block)
{
  return block.filterDomain();
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const Block &block)
{
  assert(0);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &ray, const Block &block)
{
  hit_record<Ray, primitive<unsigned>> result;
  float3 pos = ray.ori;
  result.hit = false;

  if (!block.filterDomain().contains(pos)) {
    return result;
  }

  float *prd = (float *)ray.prd;
  float &sumWeightedValues = prd[0];
  float &sumWeights = prd[1];

  const float3 P = ray.ori;
  const aabb brickBounds = block.worldBounds();
  const int3 blockSize = block.numCells();

  const float3 localPos = (P-brickBounds.min) / float3(block.cellSize()) - 0.5f;
  int3 idx_lo   = int3(floorf(localPos.x),floorf(localPos.y),floorf(localPos.z));
  idx_lo = max(int3(-1), idx_lo);
  const int3 idx_hi   = idx_lo + int3(1);
  const float3 frac     = localPos - float3(idx_lo);
  const float3 neg_frac = float3(1.f) - frac;

  // #define INV_CELL_WIDTH invCellWidth
  #define INV_CELL_WIDTH 1.f
  if (idx_lo.z >= 0 && idx_lo.z < blockSize.z) {
    if (idx_lo.y >= 0 && idx_lo.y < blockSize.y) {
      if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
        const float scalar = block.getScalar(idx_lo.x,idx_lo.y,idx_lo.z);
        const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
      if (idx_hi.x < blockSize.x) {
        const float scalar = block.getScalar(idx_hi.x,idx_lo.y,idx_lo.z);
        const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
    }
    if (idx_hi.y < blockSize.y) {
      if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
        const float scalar = block.getScalar(idx_lo.x,idx_hi.y,idx_lo.z);
        const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
      if (idx_hi.x < blockSize.x) {
        const float scalar = block.getScalar(idx_hi.x,idx_hi.y,idx_lo.z);
        const float weight = (neg_frac.z)*(frac.y)*(frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
    }
  }
    
  if (idx_hi.z < blockSize.z) {
    if (idx_lo.y >= 0 && idx_lo.y < blockSize.y) {
      if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
        const float scalar = block.getScalar(idx_lo.x,idx_lo.y,idx_hi.z);
        const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
      if (idx_hi.x < blockSize.x) {
        const float scalar = block.getScalar(idx_hi.x,idx_lo.y,idx_hi.z);
        const float weight = (frac.z)*(neg_frac.y)*(frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
    }
    if (idx_hi.y < blockSize.y) {
      if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
        const float scalar = block.getScalar(idx_lo.x,idx_hi.y,idx_hi.z);
        const float weight = (frac.z)*(frac.y)*(neg_frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
      if (idx_hi.x < blockSize.x) {
        const float scalar = block.getScalar(idx_hi.x,idx_hi.y,idx_hi.z);
        const float weight = (frac.z)*(frac.y)*(frac.x);
        sumWeights += weight;
        sumWeightedValues += weight*scalar;
      }
    }
  }

  return result;
}

// Grid accelerator to traverse spatial fields //

struct GridAccel
{
  int3 dims;
  box3 worldBounds;
  float *stepSizes; // step size to take
  box1 *valueRanges; // min/max ranges
  float *maxOpacities; // used as majorants

  VSNRAY_FUNC
  inline bool isValid() const
  {
    return dims != int3(0) && valueRanges && maxOpacities;
  }

  VSNRAY_FUNC
  inline box1 valueRange(int leafID) const
  {
    if (leafID >= 0 && valueRanges)
      return valueRanges[leafID];
    else
      return box1(-FLT_MAX, FLT_MAX);
  }

  VSNRAY_FUNC
  inline float stepSize(int leafID) const
  {
    if (leafID >= 0 && stepSizes)
      return stepSizes[leafID];
    else
      return 1.f;
  }
};

inline GridAccel createGridAccel()
{
  GridAccel accel;
  memset(&accel,0,sizeof(accel));
  accel.dims = int3(0);
  accel.worldBounds = box3f(float3(FLT_MAX),float3(-FLT_MAX));
  accel.stepSizes = nullptr;
  accel.valueRanges = nullptr;
  accel.maxOpacities = nullptr;
  return accel;
}

// Spatial Field //

struct SpatialField
{
  enum Type { StructuredRegular, Unstructured, BlockStructured, NanoVDB, Unknown, };
  Type type;
  unsigned fieldID;
  float delta;
  GridAccel gridAccel;
  mat4x3 voxelSpaceTransform;

  // Transform point in object space to voxel space
  VSNRAY_FUNC
  inline float3 pointToVoxelSpace(const float3 &object) const
  {
    mat3 rot = top_left(voxelSpaceTransform);
    vec3 trans = voxelSpaceTransform(3);
    return rot * (object + trans);
  }

  // Transform vector in object space to voxel space
  VSNRAY_FUNC
  inline float3 vectorToVoxelSpace(const float3 &object) const
  {
    mat3 rot = top_left(voxelSpaceTransform);
    return rot * object;
  }

  union {
    struct {
#ifdef WITH_CUDA
      cuda_texture_ref<float, 3> sampler;
#elif defined(WITH_HIP)
      hip_texture_ref<float, 3> sampler;
#else
      texture_ref<float, 3> sampler;
#endif
    } asStructuredRegular;
    struct {
      // Sampling BVH. This BVH is in _voxel_ space, so rays that take samples
      // must first be transformed there from world space in case these spaces
      // aren't the same!
#ifdef WITH_CUDA
      cuda_index_bvh<UElem>::bvh_ref samplingBVH;
#elif defined(WITH_HIP)
      hip_index_bvh<UElem>::bvh_ref samplingBVH;
#else
      index_bvh4<UElem>::bvh_ref samplingBVH;
#endif
    } asUnstructured;
    struct {
#ifdef WITH_CUDA
      cuda_index_bvh<Block>::bvh_ref samplingBVH;
#elif defined(WITH_HIP)
      hip_index_bvh<Block>::bvh_ref samplingBVH;
#else
      index_bvh4<Block>::bvh_ref samplingBVH;
#endif
    } asBlockStructured;
#ifdef WITH_NANOVDB
    struct {
      nanovdb::NanoGrid<float> *grid;
      tex_filter_mode filterMode;
    } asNanoVDB;
#endif
  };
};

inline SpatialField createSpatialField()
{
  SpatialField field;
  memset(&field,0,sizeof(field));
  field.fieldID = UINT_MAX;
  field.delta = 0.5f;
  return field;
}

VSNRAY_FUNC
inline bool sampleField(const SpatialField &sf, vec3 P, float &value) {
  // This assumes that P is in voxel space!
  if (sf.type == SpatialField::StructuredRegular) {
    value = tex3D(sf.asStructuredRegular.sampler,P);
    return true;
  } else if (sf.type == SpatialField::Unstructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;
#if defined(WITH_CUDA) || defined(WITH_HIP)
    auto hr = intersect(ray, sf.asUnstructured.samplingBVH);
#else
    auto hr = intersect_ray1_bvh4(ray, sf.asUnstructured.samplingBVH);
#endif

    if (!hr.hit)
      return false;

    value = hr.u; // value is stored in "u"!
    return true;
  } else if (sf.type == SpatialField::BlockStructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;

    // sumValues+sumWeightedValues
    float basisPRD[2] = {0.f,0.f};
    ray.prd = &basisPRD;

#if defined(WITH_CUDA) || defined(WITH_HIP)
    auto hr = intersect(ray, sf.asBlockStructured.samplingBVH);
#else
    auto hr = intersect_ray1_bvh4(ray, sf.asBlockStructured.samplingBVH);
#endif

    if (basisPRD[1] == 0.f)
      return false;

    value = basisPRD[0]/basisPRD[1];
    return true;
  }
#ifdef WITH_NANOVDB
  else if (sf.type == SpatialField::NanoVDB) {
    auto acc = sf.asNanoVDB.grid->getAccessor();
    if (sf.asNanoVDB.filterMode == Nearest) {
      auto smp = nanovdb::math::createSampler<0>(acc);
      value = smp(nanovdb::math::Vec3<float>(P.x,P.y,P.z));
      return true;
    } else if (sf.asNanoVDB.filterMode == Linear) {
      auto smp = nanovdb::math::createSampler<1>(acc);
      value = smp(nanovdb::math::Vec3<float>(P.x,P.y,P.z));
      return true;
    }
  }
#endif

  return false;
}

VSNRAY_FUNC
inline bool sampleGradient(const SpatialField &sf, vec3 P, float3 &value) {
  float x0=0, x1=0, y0=0, y1=0, z0=0, z1=0;
  bool b0 = sampleField(sf, P+float3{sf.delta, 0.f, 0.f}, x1);
  bool b1 = sampleField(sf, P-float3{sf.delta, 0.f, 0.f}, x0);
  bool b2 = sampleField(sf, P+float3{0.f, sf.delta, 0.f}, y1);
  bool b3 = sampleField(sf, P-float3{0.f, sf.delta, 0.f}, y0);
  bool b4 = sampleField(sf, P+float3{0.f, 0.f, sf.delta}, z1);
  bool b5 = sampleField(sf, P-float3{0.f, 0.f, sf.delta}, z0);
  if (b0 && b1 && b2 && b3 && b4 && b5) {
    value = float3{x1,y1,z1}-float3{x0,y0,z0};
    return true; // TODO
  } else {
    value = float3{0.f};
    return false;
  }
}

// Transfer functions //

struct TransferFunction1D
{
  unsigned numValues;
  box1 valueRange;
#ifdef WITH_CUDA
  cuda_texture_ref<float4, 1> sampler;
#elif defined(WITH_HIP)
  hip_texture_ref<float4, 1> sampler;
#else
  texture_ref<float4, 1> sampler;
#endif
};

VSNRAY_FUNC
inline float4 postClassify(TransferFunction1D tf, float v) {
  box1 valueRange = tf.valueRange;
  v = (v - valueRange.min) / (valueRange.max - valueRange.min);
  return tex1D(tf.sampler, v);
}

// Volume //

struct Volume
{
  enum Type { TransferFunction1D, Unknown, };
  Type type{Unknown};

  unsigned volID{UINT_MAX};
  float unitDistance;

  SpatialField field;
  union {
    struct TransferFunction1D asTransferFunction1D;
  };

  aabb bounds;
};

inline Volume createVolume()
{
  Volume vol;
  memset(&vol,0,sizeof(vol));
  vol.volID  = UINT_MAX;
  vol.bounds.invalidate();
  return vol;
}

VSNRAY_FUNC
inline aabb get_bounds(const Volume &vol)
{
  return vol.bounds;
}

inline void split_primitive(aabb &L, aabb &R, float plane, int axis, const Volume &vol)
{
  assert(0);
}

struct HitRecordVolume
{
  bool hit{false};
  float t{FLT_MAX};
  float3 albedo{0.f,0.f,0.f};
  float extinction{0.f};
  float Tr{1.f};
  int volID{-1};
  int instID{-1};
};

struct VolumePRD
{
  HitRecordVolume *hr;
  Random *rnd;
};

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(Ray ray, const Volume &vol)
{
  VolumePRD &prd = *(VolumePRD *)ray.prd;
  HitRecordVolume &hrv = *prd.hr;

  auto boxHit = intersect(ray,vol.bounds);

  hit_record<Ray, primitive<unsigned>> hr;
  hr.t = FLT_MAX;
  hr.hit = false;

  if (!boxHit.hit)
    return hr;

  if (ray.intersectionMask & Ray::VolumeBounds) {
    // we just report that we did hit the box; the user
    // is later expected to intersect the volume bounds
    // themselves to compute [t0,t1]
    hr.hit = boxHit.hit && (boxHit.tfar >= ray.tmin);
    hr.t = max(ray.tmin,boxHit.tnear);
    hr.geom_id = vol.volID;
    if (hr.t < hrv.t) {
      hrv.hit = true;
      hrv.t = hr.t;
      hrv.volID = hr.geom_id;
    }
    return hr;
  }

  Random &rnd = *prd.rnd;

  hr.geom_id = vol.volID;
  
  const auto &sf = vol.field;
  dco::GridAccel grid = sf.gridAccel;

  float3 albedo;
  float Tr{1.f};
  float extinction{0.f};
  float unitDistance = vol.unitDistance;

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {

    const float majorant = grid.isValid() ? grid.maxOpacities[leafID] : 1.f;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= (logf(1.f - rnd()) / (majorant * unitDistance));

      if (t >= t1)
        break;

      float3 P = ray.ori+ray.dir*t;
      float v = 0.f;
      if (sampleField(sf,P,v)) {
        float4 sample
            = postClassify(vol.asTransferFunction1D,v);
        albedo = sample.xyz();
        extinction = sample.w;
        float u = rnd();
        if (extinction >= u * majorant) {
          hr.hit = true;
          Tr = 0.f;
          hr.t = t;
          return false; // stop traversal
        }
      }
    }

    return true; // cont. traversal to the next spat. partition
  };

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  unitDistance = unitDistance * dt_scale;

  hr.t = ray.tmax;
  if (sf.gridAccel.isValid())
    dda3(ray, grid.dims, grid.worldBounds, woodcockFunc);
  else
    woodcockFunc(-1, ray.tmin, ray.tmax);

  if (hr.hit) {
    hr.t /= dt_scale;

    if (hr.t < hrv.t) {
      hrv.hit = true;
      hrv.t = hr.t;
      hrv.volID = hr.geom_id;
      hrv.albedo = albedo;
      hrv.Tr = Tr;
      hrv.extinction = extinction;
    }
  }

  return hr;
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

  const auto &sf = iso.field;

  float unitDistance = 1.f;

  auto isectFunc = [&](const int leafID, float t0, float t1) {
    bool empty = (leafID != -1);

    float dt = unitDistance * sf.gridAccel.stepSize(leafID);
    box1 valueRange = sf.gridAccel.valueRange(leafID);

    for (unsigned i=0;i<iso.numValues;i++) {
      float isoValue = iso.values[i];
      if (valueRange.contains(isoValue)) {
        empty = false;
        break;
      }
    }

    if (empty)
      return true;

    float t0_old = t0;
    float t1_old = t1;
    t0 = t1 = ray.tmin-dt/2.f;
    while (t0 < t0_old) t0 += dt;
    while (t1 < t1_old) t1 += dt;

    float3 P1 = ray.ori+ray.dir*t0;
    float v1 = 0.f;
    bool sample1 = sampleField(sf,P1,v1);

    for (float t=t0;t<t1;t+=dt) {
      float3 P2 = ray.ori+ray.dir*(t+dt);
      float v2 = 0.f;
      bool sample2 = sampleField(sf,P2,v2);
      if (sample1 && sample2) {
        box1f ival(fminf(v1,v2), fmaxf(v1,v2));
        unsigned numISOs = iso.numValues;
        bool hit=false;
        for (unsigned i=0;i<numISOs;i++) {
          float isoValue = iso.values[i];
          if (ival.contains(isoValue)) {
            //float tHit = t+dt/2.f;
            float f = (isoValue-v1) / (v2-v1);
            float tHit = t+dt*f;
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
      P1 = P2;
      v1 = v2;
      sample1 = sample2;
    }

    return true; // cont. traversal to the next spat. partition
  };

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  unitDistance = unitDistance * dt_scale;

  if (sf.gridAccel.isValid())
    dda3(ray, sf.gridAccel.dims, sf.gridAccel.worldBounds, isectFunc);
  else
    isectFunc(-1, boxHit.tnear, boxHit.tfar);

  if (result.hit) {
    result.t /= dt_scale;
  }

  return result;
}

// Cone primitive //

struct Cone : public primitive<unsigned>
{
  float3 v1, v2;
  float r1, r2;
};

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &r, const Cone &cone)
{
  // From https://iquilezles.org/articles/intersectors/
  hit_record<Ray, primitive<unsigned>> result;
  result.hit = false;

  const vec3f &ro = r.ori;
  const vec3f &rd = r.dir;

  const vec3f &pa = cone.v1;
  const vec3f &pb = cone.v2;

  const float ra = cone.r1;
  const float rb = cone.r2;

  const vec3f ba = pb - pa;
  const vec3f oa = ro - pa;
  const vec3f ob = ro - pb;
  const float m0 = dot(ba,ba);
  const float m1 = dot(oa,ba);
  const float m2 = dot(rd,ba);
  const float m3 = dot(rd,oa);
  const float m5 = dot(oa,oa);
  const float m9 = dot(ob,ba);

  auto dot2 = [](const vec3f v) { return dot(v,v); };

  // Caps:
  if (m1 < 0.f) {
    if (dot2(oa*m2-rd*m1) < ra*ra*m2*m2) {
      result.t = -m1 / m2;
      result.u = 0.f;
      result.hit = true;
      result.isect_pos = r.ori + result.t * r.dir;
      result.prim_id = cone.prim_id;
      result.geom_id = cone.geom_id;
      return result;
    }
  } else if (m9 > 0.f) {
    const float t = -m9/m2;
    if (dot2(ob+rd*t) < rb*rb) {
      result.t = t;
      result.u = 1.f;
      result.hit = true;
      result.isect_pos = r.ori + result.t * r.dir;
      result.prim_id = cone.prim_id;
      result.geom_id = cone.geom_id;
      return result;
    }
  }

  // Body
  const float rr = ra - rb;
  const float hy = m0 + rr*rr;
  const float k2 = m0*m0 - m2*m2*hy;
  const float k1 = m0*m0*m3 - m1*m2*hy + m0*ra*(rr*m2*1.f);
  const float k0 = m0*m0*m5 - m1*m1*hy + m0*ra*(rr*m1*2.f - m0*ra);
  const float h = k1*k1 - k2*k0;
  if (h < 0.f) return result;
  const float t = (-k1-sqrtf(h))/k2;
  const float y = m1 + t*m2;

  if (y > 0.f && y<m0) {
    result.t = t;
    result.u = y/m0;
    result.v = y;
    result.hit = true;
    result.isect_pos = r.ori + result.t * r.dir;
    result.prim_id = cone.prim_id;
    result.geom_id = cone.geom_id;
  }

  return result;
}

VSNRAY_FUNC inline aabb get_bounds(const Cone &cone)
{
  aabb result;
  result.invalidate();
  result.insert(cone.v1 - cone.r1);
  result.insert(cone.v1 + cone.r1);
  result.insert(cone.v2 - cone.r2);
  result.insert(cone.v2 + cone.r2);
  return result;
}

VSNRAY_FUNC inline void split_primitive(
    aabb& L, aabb& R, float plane, int axis, const Cone &cone)
{
  VSNRAY_UNUSED(L);
  VSNRAY_UNUSED(R);
  VSNRAY_UNUSED(plane);
  VSNRAY_UNUSED(axis);
  VSNRAY_UNUSED(cone);

  // TODO: implement this to support SBVHs
}

// Bezier curve primitive //

struct BezierCurve : public primitive<unsigned>
{
  float3 w0, w1, w2, w3;
  float r;

  VSNRAY_FUNC vec3 f(float t) const
  {
    float tinv = 1.0f - t;
    return tinv * tinv * tinv * w0
     + 3.0f * tinv * tinv * t * w1
        + 3.0f * tinv * t * t * w2
                  + t * t * t * w3;
  }

  VSNRAY_FUNC vec3 dfdt(float t) const
  {
    float tinv = 1.0f - t;
    return                 -3.0f * tinv * tinv * w0
     + 3.0f * (3.0f * t * t - 4.0f * t + 1.0f) * w1
                + 3.0f * (2.0f - 3.0f * t) * t * w2
                                + 3.0f * t * t * w3;
  }
};

VSNRAY_FUNC
inline BezierCurve make_bezierCurve(
    const vec3 &w0, const vec3 &w1, const vec3 &w2, const vec3 &w3, float r)
{
  BezierCurve curve;
  curve.w0 = w0;
  curve.w1 = w1;
  curve.w2 = w2;
  curve.w3 = w3;
  curve.r = r;
  return curve;
}

//=========================================================
// Phantom Ray-Hair Intersector (Reshetov and Luebke, 2018)
//=========================================================

namespace phantom {

// Ray/cone intersection from appendix A

struct RayConeIntersection
{
  VSNRAY_FUNC inline bool intersect(float r, float dr)
  {
    float r2  = r * r;
    float drr = r * dr;

    float ddd = cd.x * cd.x + cd.y * cd.y;
    dp        = c0.x * c0.x + c0.y * c0.y;
    float cdd = c0.x * cd.x + c0.y * cd.y;
    float cxd = c0.x * cd.y - c0.y * cd.x;

    float c = ddd;
    float b = cd.z * (drr - cdd);
    float cdz2 = cd.z * cd.z;
    ddd += cdz2;
    float a = 2.0f * drr * cdd + cxd * cxd - ddd * r2 + dp * cdz2;

    float discr = b * b - a * c;
    s   = (b - (discr > 0.0f ? sqrtf(discr) : 0.0f)) / c;
    dt  = (s * cd.z - cdd) / ddd;
    dc  = s * s + dp;
    sp  = cdd / cd.z;
    dp += sp * sp;

    return discr > 0.0f;
  }

  vec3  c0;
  vec3  cd;
  float s;
  float dt;
  float dp;
  float dc;
  float sp;
};

// TODO: use visionaray's ray/cyl test?!
VSNRAY_FUNC inline
bool intersectCylinder(const Ray &ray, vec3 p0, vec3 p1, float ra)
{
  vec3  ba = p1 - p0;
  vec3  oc = ray.ori - p0;

  float baba = dot(ba, ba);
  float bard = dot(ba, ray.dir);
  float baoc = dot(ba, oc);

  float k2 = baba - bard * bard;
  float k1 = baba * dot(oc, ray.dir) - baoc * bard;
  float k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;

  float h = k1 * k1 - k2 * k0;

  if (h < 0.0f)
    return false;

  h = sqrtf(h);
  float t = (-k1 - h) / k2;

  // body
  float y = baoc + t * bard;
  if (y > 0.0f && y < baba)
    return true;

  // caps
  t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;
  if (fabsf(k1 + k2 * t) < h)
    return true;

  return false;
}

struct TransformToRCC
{
  VSNRAY_FUNC inline TransformToRCC(const Ray &r)
  {
    vec3 e1;
    vec3 e2;
    vec3 e3 = normalize(r.dir);
    make_orthonormal_basis(e1, e2, e3);
    xformInv = mat4(
        vec4(e1,    0.0f),
        vec4(e2,    0.0f),
        vec4(e3,    0.0f),
        vec4(r.ori, 1.0f)
        );
    xform = inverse(xformInv);
  }

  VSNRAY_FUNC inline vec3 xfmPoint(vec3 point)
  { return (xform * vec4(point, 1.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmVector(vec3 vector)
  { return (xform * vec4(vector, 0.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmPointInv(vec3 point)
  { return (xformInv * vec4(point, 1.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmVectorInv(vec3 vector)
  { return (xformInv * vec4(vector, 0.0f)).xyz(); }

  mat4 xform;
  mat4 xformInv;
};

} // namespace phantom

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &r, const BezierCurve &curve)
{
  hit_record<Ray, primitive<unsigned>> result;
  result.hit = false;

  // Early exit check against enclosing cylinder
  auto distToCylinder = [&curve](vec3 pt) {
    return length(cross(pt - curve.w0, pt - curve.w3)) / length(curve.w3 - curve.w0);
  };

  // TODO: could compute tighter bounding cylinder than this one!
  float rmax = distToCylinder(curve.f(0.33333f));
  rmax = fmaxf(rmax, distToCylinder(curve.f(0.66667f)));
  rmax += curve.r;

  vec3 axis = normalize(curve.w3 - curve.w0);
  vec3 p0   = curve.w0 - axis * curve.r;
  vec3 p1   = curve.w3 + axis * curve.r;

  if (!phantom::intersectCylinder(r, p0, p1, rmax))
    return result;

  // Transform curve to RCC
  phantom::TransformToRCC rcc(r);
  BezierCurve xcurve = make_bezierCurve(
      rcc.xfmPoint(curve.w0),
      rcc.xfmPoint(curve.w1),
      rcc.xfmPoint(curve.w2),
      rcc.xfmPoint(curve.w3),
      curve.r
      );

  // "Test for convergence. If the intersection is found,
  // report it, otherwise start at the other endpoint."

  // Compute curve end to start at
  float tstart = dot(xcurve.w3 - xcurve.w0, r.dir) > 0.0f ? 0.0f : 1.0f;

  for (int ep = 0; ep < 2; ++ep)
  {
    float t   = tstart;

    phantom::RayConeIntersection rci;

    float told = 0.0f;
    float dt1 = 0.0f;
    float dt2 = 0.0f;

    for (int i = 0; i < 40; ++i)
    {
      rci.c0 = xcurve.f(t);
      rci.cd = xcurve.dfdt(t);

      bool phantom = !rci.intersect(curve.r, 0.0f/*cylinder*/);

      // "In all examples in this paper we stop iterations when dt < 5x10^−5"
      if (!phantom && fabsf(rci.dt) < 5e-5f) {
        //vec3 n = normalize(curve.dfdt(t));
        rci.s += rci.c0.z;
        result.t = rci.s;
        result.u = t; // abuse param u to store curve's t
        result.hit = true;
        result.isect_pos = r.ori + result.t * r.dir;
        break;
      }

      rci.dt = min(rci.dt, 0.5f);
      rci.dt = max(rci.dt, -0.5f);

      dt1 = dt2;
      dt2 = rci.dt;

      // Regula falsi
      if (dt1 * dt2 < 0.0f) {
        float tnext = 0.0f;
        // "we use the simplest possible approach by switching
        // to the bisection every 4th iteration:"
        if ((i & 3) == 0)
          tnext = 0.5f * (told + t);
        else
          tnext = (dt2 * told - dt1 * t) / (dt2 - dt1);
        told = t;
        t = tnext;
      } else {
        told = t;
        t += rci.dt;
      }

      if (t < 0.0f || t > 1.0f)
        break;
    }

    if (!result.hit)
      tstart = 1.0f - tstart;
    else
      break;
  }

  return result;
}

// From here: https://www.shadertoy.com/view/MdKBWt
VSNRAY_FUNC inline aabb get_bounds(const BezierCurve &curve)
{
  vec3 p0 = curve.w0;
  vec3 p1 = curve.w1;
  vec3 p2 = curve.w2;
  vec3 p3 = curve.w3;

  // extremes
  vec3 mi = min(p0,p3);
  vec3 ma = max(p0,p3);

  // note pascal triangle coefficnets
  vec3 c = -1.0f*p0 + 1.0f*p1;
  vec3 b =  1.0f*p0 - 2.0f*p1 + 1.0f*p2;
  vec3 a = -1.0f*p0 + 3.0f*p1 - 3.0f*p2 + 1.0f*p3;

  // check if curve is quadratic, then derivative is a line.
  // in that case we'll just lazily insert the remaining control points..
  for (int d=0; d<3; ++d) {
    if (a[d] == 0.f) {
      mi[d] = min(mi[d],p1[d]);
      mi[d] = min(mi[d],p2[d]);
      ma[d] = max(mi[d],p1[d]);
      ma[d] = max(mi[d],p2[d]);
    }
  }

  vec3 h = b*b - a*c;

  // real solutions
  if (h.x > 0.0f || h.y > 0.0f || h.z > 0.0f)
  {
    vec3 g(sqrtf(fabsf(h.x)), sqrtf(fabsf(h.y)), sqrtf(fabsf(h.z)));
    vec3 t1 = clamp((-b - g)/a,vec3(0.0f),vec3(1.0f)); vec3 s1 = 1.0f-t1;
    vec3 t2 = clamp((-b + g)/a,vec3(0.0f),vec3(1.0f)); vec3 s2 = 1.0f-t2;
    vec3 q1 = s1*s1*s1*p0 + 3.0f*s1*s1*t1*p1 + 3.0f*s1*t1*t1*p2 + t1*t1*t1*p3;
    vec3 q2 = s2*s2*s2*p0 + 3.0f*s2*s2*t2*p1 + 3.0f*s2*t2*t2*p2 + t2*t2*t2*p3;

    if (h.x > 0.0f) {
      mi.x = min(mi.x,min(q1.x,q2.x));
      ma.x = max(ma.x,max(q1.x,q2.x));
    }

    if (h.y > 0.0f) {
      mi.y = min(mi.y,min(q1.y,q2.y));
      ma.y = max(ma.y,max(q1.y,q2.y));
    }

    if (h.z > 0.0f) {
      mi.z = min(mi.z,min(q1.z,q2.z));
      ma.z = max(ma.z,max(q1.z,q2.z));
    }
  }

  return aabb(mi - vec3(curve.r), ma + vec3(curve.r));
}

VSNRAY_FUNC inline void split_primitive(
    aabb& L, aabb& R, float plane, int axis, const BezierCurve &curve)
{
  VSNRAY_UNUSED(L);
  VSNRAY_UNUSED(R);
  VSNRAY_UNUSED(plane);
  VSNRAY_UNUSED(axis);
  VSNRAY_UNUSED(curve);

  // TODO: implement this to support SBVHs
}

// BLS primitives //

struct BLS
{
  enum Type {
    Triangle,
    Quad,
    Sphere,
    Cone,
    Cylinder,
    Curve,
    BezierCurve,
    ISOSurface,
    Volume,
    Unknown,
  };
  Type type{Unknown};
  unsigned blsID{UINT_MAX};
#ifdef WITH_CUDA
  union {
    cuda_index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
    cuda_index_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
    cuda_index_bvh<basic_sphere<float>>::bvh_ref asSphere;
    cuda_index_bvh<dco::Cone>::bvh_ref asCone;
    cuda_index_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
    cuda_index_bvh<dco::BezierCurve>::bvh_ref asBezierCurve;
    cuda_index_bvh<dco::ISOSurface>::bvh_ref asISOSurface;
    cuda_index_bvh<dco::Volume>::bvh_ref asVolume;
  };
#elif defined(WITH_HIP)
  union {
    hip_index_bvh<basic_triangle<3,float>>::bvh_ref asTriangle;
    hip_index_bvh<basic_triangle<3,float>>::bvh_ref asQuad;
    hip_index_bvh<basic_sphere<float>>::bvh_ref asSphere;
    hip_index_bvh<dco::Cone>::bvh_ref asCone;
    hip_index_bvh<basic_cylinder<float>>::bvh_ref asCylinder;
    hip_index_bvh<dco::BezierCurve>::bvh_ref asBezierCurve;
    hip_index_bvh<dco::ISOSurface>::bvh_ref asISOSurface;
    hip_index_bvh<dco::Volume>::bvh_ref asVolume;
  };
#else
  union {
    index_bvh4<basic_triangle<3,float>>::bvh_ref asTriangle;
    index_bvh4<basic_triangle<3,float>>::bvh_ref asQuad;
    index_bvh4<basic_sphere<float>>::bvh_ref asSphere;
    index_bvh4<dco::Cone>::bvh_ref asCone;
    index_bvh4<basic_cylinder<float>>::bvh_ref asCylinder;
    index_bvh4<dco::BezierCurve>::bvh_ref asBezierCurve;
    index_bvh4<dco::ISOSurface>::bvh_ref asISOSurface;
    index_bvh4<dco::Volume>::bvh_ref asVolume;
  };
#endif
};

VSNRAY_FUNC
inline aabb get_bounds(const BLS &bls)
{
#ifdef WITH_HIP
  // with HIP we currenlty assume that TLSs are built on the host:
  bvh_node hip_root;
  if (bls.type == BLS::Triangle && bls.asTriangle.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asTriangle.nodes(), sizeof(hip_root), hipMemcpyDefault));
  if (bls.type == BLS::Quad && bls.asQuad.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asQuad.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Sphere && bls.asSphere.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asSphere.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Cone && bls.asCone.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asCone.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Cylinder && bls.asCylinder.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asCylinder.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::BezierCurve && bls.asBezierCurve.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asBezierCurve.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::ISOSurface && bls.asISOSurface.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asISOSurface.nodes(), sizeof(hip_root), hipMemcpyDefault));
  else if (bls.type == BLS::Volume && bls.asVolume.num_nodes())
    HIP_SAFE_CALL(hipMemcpy(
        &hip_root, bls.asVolume.nodes(), sizeof(hip_root), hipMemcpyDefault));
  return hip_root.get_bounds();
#else
  if (bls.type == BLS::Triangle && bls.asTriangle.num_nodes())
    return bls.asTriangle.node(0).get_bounds();
  if (bls.type == BLS::Quad && bls.asQuad.num_nodes())
    return bls.asQuad.node(0).get_bounds();
  else if (bls.type == BLS::Sphere && bls.asSphere.num_nodes())
    return bls.asSphere.node(0).get_bounds();
  else if (bls.type == BLS::Cone && bls.asCone.num_nodes())
    return bls.asCone.node(0).get_bounds();
  else if (bls.type == BLS::Cylinder && bls.asCylinder.num_nodes())
    return bls.asCylinder.node(0).get_bounds();
  else if (bls.type == BLS::BezierCurve && bls.asBezierCurve.num_nodes())
    return bls.asBezierCurve.node(0).get_bounds();
  else if (bls.type == BLS::ISOSurface && bls.asISOSurface.num_nodes())
    return bls.asISOSurface.node(0).get_bounds();
  else if (bls.type == BLS::Volume && bls.asVolume.num_nodes())
    return bls.asVolume.node(0).get_bounds();
#endif

  aabb inval;
  inval.invalidate();
  return inval;
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(const Ray &ray, const BLS &bls)
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  if (bls.type == BLS::Triangle && (ray.intersectionMask & Ray::Triangle))
    return intersect(ray,bls.asTriangle);
  else if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    return intersect(ray,bls.asQuad);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    return intersect(ray,bls.asSphere);
  else if (bls.type == BLS::Cone && (ray.intersectionMask & Ray::Cone))
    return intersect(ray,bls.asCone);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    return intersect(ray,bls.asCylinder);
  else if (bls.type == BLS::BezierCurve && (ray.intersectionMask & Ray::BezierCurve))
    return intersect(ray,bls.asBezierCurve);
  else if (bls.type == BLS::ISOSurface && (ray.intersectionMask & Ray::ISOSurface))
    return intersect(ray,bls.asISOSurface);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::Volume))
    return intersect(ray,bls.asVolume);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::VolumeBounds))
    return intersect(ray,bls.asVolume);
#else
  if (bls.type == BLS::Triangle && (ray.intersectionMask & Ray::Triangle))
    return intersect_ray1_bvh4(ray,bls.asTriangle);
  else if (bls.type == BLS::Quad && (ray.intersectionMask & Ray::Quad))
    return intersect_ray1_bvh4(ray,bls.asQuad);
  else if (bls.type == BLS::Sphere && (ray.intersectionMask & Ray::Sphere))
    return intersect_ray1_bvh4(ray,bls.asSphere);
  else if (bls.type == BLS::Cone && (ray.intersectionMask & Ray::Cone))
    return intersect_ray1_bvh4(ray,bls.asCone);
  else if (bls.type == BLS::Cylinder && (ray.intersectionMask & Ray::Cylinder))
    return intersect_ray1_bvh4(ray,bls.asCylinder);
  else if (bls.type == BLS::BezierCurve && (ray.intersectionMask & Ray::BezierCurve))
    return intersect_ray1_bvh4(ray,bls.asBezierCurve);
  else if (bls.type == BLS::ISOSurface && (ray.intersectionMask & Ray::ISOSurface))
    return intersect_ray1_bvh4(ray,bls.asISOSurface);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::Volume))
    return intersect_ray1_bvh4(ray,bls.asVolume);
  else if (bls.type == BLS::Volume && (ray.intersectionMask & Ray::VolumeBounds))
    return intersect_ray1_bvh4(ray,bls.asVolume);
#endif

  return {};
}

// Array //

struct Array
{
  const void *data;
  size_t len;
  TypeInfo typeInfo;
};

enum class Attribute
{
  _0, _1, _2, _3, Color, None,
};

// Instance //

struct Instance
{
  enum Type { Transform, MotionTransform, Unknown, };
  Type type;
  unsigned instID;
  unsigned userID;
  unsigned groupID;
#ifdef WITH_CUDA
  cuda_index_bvh<BLS>::bvh_ref theBVH;
#elif defined(WITH_HIP)
  hip_index_bvh<BLS>::bvh_ref theBVH;
#else
  index_bvh<BLS>::bvh_ref theBVH;
#endif
  mat4 *xfms;
  mat3 *normalXfms;
  mat3 *affineInv;
  vec3 *transInv;
  unsigned len;
  box1 time;
};

inline Instance createInstance()
{
  Instance inst;
  memset(&inst,0,sizeof(inst));
  inst.type    = Instance::Unknown;
  inst.instID  = UINT_MAX;
  inst.userID  = UINT_MAX;
  inst.groupID = UINT_MAX;
  return inst;
}

VSNRAY_FUNC
inline aabb get_bounds(const Instance &inst)
{
  if (inst.type == Instance::Transform && inst.theBVH.num_nodes()) {

    aabb bound = inst.theBVH.node(0).get_bounds();
    mat3f rot = inverse(inst.affineInv[0]);
    vec3f trans = -inst.transInv[0];
    auto verts = compute_vertices(bound);
    aabb result;
    result.invalidate();
    for (vec3 v : verts) {
      v = rot * v + trans;
      result.insert(v);
    }
    return result;
  } else if (inst.type == Instance::MotionTransform && inst.len) {
    aabb result;
    result.invalidate();
    for (unsigned i = 0; i < inst.len; ++i) {
      aabb bound = inst.theBVH.node(0).get_bounds();
      mat3f rot = inverse(inst.affineInv[i]);
      vec3f trans = -inst.transInv[i];
      auto verts = compute_vertices(bound);
      for (vec3 v : verts) {
        v = rot * v + trans;
        result.insert(v);
      }
    }
    return result;
  }

  return {};
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &ray, const Instance &inst)
{
  mat3 affineInv;
  vec3 transInv;

  if (inst.type == Instance::Transform) {
    affineInv = inst.affineInv[0];
    transInv = inst.transInv[0];
  } else if (inst.type == Instance::MotionTransform) {
    float rayTime = clamp(ray.time, inst.time.min, inst.time.max);

    float time01 = rayTime - inst.time.min / (inst.time.max - inst.time.min);

    unsigned ID1 = unsigned(float(inst.len-1) * time01);
    unsigned ID2 = min(inst.len-1, ID1+1);

    float frac = time01 * (inst.len-1) - ID1;

    affineInv = lerp(inst.affineInv[ID1],
                     inst.affineInv[ID2],
                     frac);

    transInv = lerp(inst.transInv[ID1],
                    inst.transInv[ID2],
                    frac);
  }

  Ray xfmRay(ray);
  xfmRay.ori = affineInv * (xfmRay.ori + transInv);
  xfmRay.dir = affineInv * xfmRay.dir;

  auto hr = intersect(xfmRay,inst.theBVH);
  if (hr.hit) {
    hr.isect_pos = xfmRay.ori + hr.t * xfmRay.dir;
    hr.inst_id = inst.instID;
  } else {
    hr.inst_id = ~0u;
  }
  return hr;
}

// TLS //

#ifdef WITH_CUDA
typedef cuda_index_bvh<Instance>::bvh_ref TLS;
#elif defined(WITH_HIP)
typedef hip_index_bvh<Instance>::bvh_ref TLS;
#else
typedef index_bvh<Instance>::bvh_ref TLS;
#endif

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    Ray ray, const TLS &tls)
{
  ray.intersectionMask
      = Ray::Triangle | Ray::Quad | Ray::Sphere | Ray::Cone | Ray::Cylinder |
        Ray::Curve | Ray::BezierCurve | Ray::ISOSurface;
  return intersect(ray, tls);
}

VSNRAY_FUNC
inline HitRecordVolume intersectVolumeBounds(Ray ray, const TLS &tls)
{
  HitRecordVolume result;

  VolumePRD prd;
  prd.hr = &result;
  ray.prd = &prd;

  ray.intersectionMask = Ray::VolumeBounds;

  auto hr = intersect(ray, tls);

  result.instID = hr.inst_id;

  return result;
}

VSNRAY_FUNC
inline HitRecordVolume intersectVolumes(Ray ray, const TLS &tls)
{
  HitRecordVolume result;

  VolumePRD prd;
  prd.hr = &result;
  prd.rnd = (Random *)ray.prd;
  ray.prd = &prd;

  ray.intersectionMask = Ray::Volume;
  auto hr = intersect(ray, tls);

  result.instID = hr.inst_id;

  return result;
}

// Surface //

struct Surface
{
  unsigned surfID;
  unsigned geomID;
  unsigned matID;
};

inline Surface createSurface()
{
  Surface surf;
  surf.surfID = UINT_MAX;
  surf.geomID = UINT_MAX;
  surf.matID = UINT_MAX;
  return surf;
}

// Geometry types (for dispatch) //

using Triangle = basic_triangle<3,float>;
using Sphere = basic_sphere<float>;
using Cylinder = basic_cylinder<float>;

// Geometry //

struct Geometry
{
  enum Type {
    Triangle,
    Quad,
    Sphere,
    Cone,
    Cylinder,
    Curve,
    BezierCurve,
    ISOSurface,
    Unknown,
  };
  Type type;
  unsigned geomID;

  template <typename Primitive>
  VSNRAY_FUNC
  Primitive &as(unsigned primID)
  {
    return ((Primitive *)primitives.data)[primID];
  }

  template <typename Primitive>
  VSNRAY_FUNC
  const Primitive &as(unsigned primID) const
  {
    return ((const Primitive *)primitives.data)[primID];
  }

  Array primitives;
  Array primitiveAttributes[5];
  Array vertexAttributes[5];
  Array index;
  Array normal;
  Array tangent;
};

inline Geometry createGeometry()
{
  Geometry geom;
  memset(&geom,0,sizeof(geom));
  geom.type = Geometry::Unknown;
  geom.geomID = UINT_MAX;
  return geom;
}

VSNRAY_FUNC
inline aabb get_bounds(const Geometry &geom)
{
  aabb result;
  result.invalidate();

  if (geom.type == dco::Geometry::Triangle) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Triangle>(i)));
    }
  } else if (geom.type == dco::Geometry::Quad) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Triangle>(i)));
    }
  } else if (geom.type == dco::Geometry::Sphere) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Sphere>(i)));
    }
  } else if (geom.type == dco::Geometry::Cone) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Cone>(i)));
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Cylinder>(i)));
    }
  } else if (geom.type == dco::Geometry::BezierCurve) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::BezierCurve>(i)));
    }
  } else if (geom.type == dco::Geometry::ISOSurface) {
    if (geom.primitives.len) {
      result.insert(get_bounds(geom.as<dco::ISOSurface>(0)));
    }
  }

  return result;
}

// Sampler //

struct Sampler
{
  enum Type { Image1D, Image2D, Image3D, Transform, Primitive, Unknown, };
  Type type;
  unsigned samplerID;
  Attribute inAttribute;
  mat4 inTransform;
  float4 inOffset;
  mat4 outTransform;
  float4 outOffset;
  union {
#ifdef WITH_CUDA
    cuda_texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    cuda_texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    cuda_texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#elif defined(WITH_HIP)
    hip_texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    hip_texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    hip_texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#else
    texture_ref<vector<4, unorm<8>>, 1> asImage1D;
    texture_ref<vector<4, unorm<8>>, 2> asImage2D;
    texture_ref<vector<4, unorm<8>>, 3> asImage3D;
#endif
    struct {
      TypeInfo typeInfo;
      size_t len; // in elements
      const uint8_t *data;
      uint32_t offset;
    } asPrimitive;
  };

  VSNRAY_FUNC
  bool isValid() const
  {
    return samplerID < UINT_MAX &&
        inAttribute != Attribute::None &&
        (type == Image1D && asImage1D) ||
        (type == Image2D && asImage2D) ||
        (type == Image3D && asImage3D) ||
        (type == Primitive && asPrimitive.data) ||
        (type == Transform);
  }
};

inline Sampler createSampler()
{
  Sampler samp;
  memset(&samp,0,sizeof(samp));
  samp.type = Sampler::Unknown;
  samp.samplerID = UINT_MAX;
  samp.inAttribute = Attribute::_0;
  samp.inTransform = mat4::identity();
  samp.inOffset = float4(0.f);
  samp.outTransform = mat4::identity();
  samp.outOffset = float4(0.f);
  return samp;
}

// Params used by materials //

struct MaterialParamRGB
{
  float3 rgb;
  unsigned samplerID;
  Attribute attribute;
};

struct MaterialParamF
{
  float f;
  unsigned samplerID;
  Attribute attribute;
};

enum class AlphaMode
{
  Opaque, Blend, Mask,
};

// Material //

struct Material
{
  enum Type { Matte, PhysicallyBased, Unknown, };
  Type type;
  unsigned matID;
  union {
    struct {
      MaterialParamRGB color;
      MaterialParamF opacity;
      AlphaMode alphaMode;
      float alphaCutoff;
    } asMatte;
    struct {
      MaterialParamRGB baseColor;
      MaterialParamF opacity;
      MaterialParamF metallic;
      MaterialParamF roughness;
      struct {
        unsigned samplerID;
      } normal;
      AlphaMode alphaMode;
      float alphaCutoff;
      MaterialParamF clearcoat;
      MaterialParamF clearcoatRoughness;
      float ior;
    } asPhysicallyBased;
  };
};

inline Material createMaterial()
{
  Material mat;
  memset(&mat,0,sizeof(mat));
  mat.type = Material::Unknown;
  mat.matID = UINT_MAX;
  return mat;
};

VSNRAY_FUNC
inline Material makeDefaultMaterial()
{
  Material mat;
  mat.type = Material::Matte;
  mat.asMatte.color.rgb = vec3(0,1,0);
  mat.asMatte.color.samplerID = UINT_MAX;
  mat.asMatte.color.attribute = dco::Attribute::None;
  mat.asMatte.opacity.f = 1.f;
  mat.asMatte.opacity.samplerID = UINT_MAX;
  mat.asMatte.opacity.attribute = dco::Attribute::None;
  mat.asMatte.alphaMode = AlphaMode::Opaque;
  mat.asMatte.alphaCutoff = 0.5f;
  return mat;
}

// Quad (light) //
/// TODO: this geom type is at the moment only used for quad _lights_,
/// but could just as easily be used for the quad _primitive_ where
/// we currently just use two visionaray triangles

struct Quad
{
  vec3 v1, e1, e2;

  VSNRAY_FUNC
  inline void tessellate(
      basic_triangle<3,float> &t1, basic_triangle<3,float> &t2) const {
    t1.v1 = v1; t1.e1 = e1; t1.e2 = e1+e2;
    t2.v1 = v1; t2.e1 = e1+e2; t2.e2 = e2;
  }
};

VSNRAY_FUNC
inline aabb get_bounds(const Quad &q)
{
  aabb bounds;
  bounds.invalidate();
  bounds.insert(q.v1);
  bounds.insert(q.v1+q.e1);
  bounds.insert(q.v1+q.e2);
  bounds.insert(q.v1+q.e1+q.e2);
  return bounds;
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(const Ray &ray, const Quad &q)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);

  auto hr1 = intersect(ray, t1);
  auto hr2 = intersect(ray, t2);

  hit_record<Ray, primitive<unsigned>> result;

  if (hr1.hit) {
    result.hit = true;
    result.t = hr1.t;
  }

  if (hr2.hit && hr2.t < result.t) {
    result.hit = true;
    result.t = hr2.t;
  }

  if (result.hit)
    result.isect_pos = ray.ori + ray.dir * result.t;

  return result;
}

template <typename HR>
VSNRAY_FUNC
inline vec3 get_normal(const HR &hr, const Quad &q)
{
  (void)hr;
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);
  return normalize(cross(t1.e1,t1.e2));
  //return normalize(cross(q.e1,q.e2));
}

VSNRAY_FUNC
inline float area(const Quad &q)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);
  return area(t1) + area(t2);
}

template <typename RNG>
VSNRAY_FUNC
inline vec3 sample_surface(const Quad &q, const vec3 reference_point, RNG &rng)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);

  float A1 = area(t1);
  float A2 = area(t2);

  float r = rng();
  if (A1/(A1+A2) < r)
    return sample_surface(t1, reference_point, rng);
  else
    return sample_surface(t2, reference_point, rng);
}

// Light //

struct Light
{
  enum Type { Directional, Point, Quad, Spot, HDRI, Unknown, };
  Type type;
  unsigned lightID;
  bool visible;
  union {
    directional_light<float> asDirectional;
    point_light<float> asPoint;
    spot_light<float> asSpot;
    area_light<float,dco::Quad> asQuad;
    struct {
#ifdef WITH_CUDA
      cuda_texture_ref<float4, 2> radiance;
#elif defined(WITH_HIP)
      hip_texture_ref<float4, 2> radiance;
#else
      texture_ref<float4, 2> radiance;
#endif
      float scale;
      struct CDF {
        float *rows;
        float *lastCol;
        unsigned width;
        unsigned height;
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
        return tex2D(radiance, toUV(dir)).xyz()*scale;
      }

    } asHDRI;
  };
};

inline Light createLight()
{
  Light light;
  memset(&light,0,sizeof(light));
  light.type = Light::Unknown;
  light.lightID = UINT_MAX;
  light.visible = true;
  return light;
}

// Group //

struct Group
{
  unsigned groupID;

  unsigned numBLSs;
  dco::BLS *BLSs;
  unsigned numGeoms;
  Handle *geoms;
  unsigned numMaterials;
  Handle *materials;
  unsigned numVolumes;
  Handle *volumes;
  unsigned numLights;
  Handle *lights;
  uint32_t *objIds; // surface IDs, volume IDs, etc.
  unsigned numObjIds;
};

inline Group createGroup()
{
  Group group;
  memset(&group,0,sizeof(group));
  group.groupID = UINT_MAX;
  return group;
}

// World //

struct World
{
  unsigned worldID;

  unsigned numLights;
  // flat list of lights active in all groups:
  Handle *allLights;
};

inline World createWorld()
{
  World world;
  world.worldID = UINT_MAX;
  world.numLights = 0;
  world.allLights = nullptr;
  return world;
}

// Camera //

struct Camera
{
  enum Type { Matrix, Pinhole, Ortho, Unknown, };
  Type type;
  unsigned camID;
  box1 shutter;
  thin_lens_camera asPinholeCam;
  union {
    matrix_camera asMatrixCam;
    struct {
      void init(float3 pos, float3 dir, float3 up, float aspect, float height,
                box2f image_region)
      {
        this->pos = pos;
        this->dir = dir;
        this->up  = up;
        this->image_region = image_region;

        float2 imgPlaneSize(height * aspect, height);

        U = normalize(cross(dir, up)) * imgPlaneSize.x;
        V = normalize(cross(U, dir)) * imgPlaneSize.y;
        W = pos - 0.5f * U - 0.5f * V;
      }

      VSNRAY_FUNC
      inline Ray primary_ray(Ray/**/, float x, float y, float width, float height) const
      {
        float2 screen((x + 0.5f) / width, (y + 0.5f) / height);
        screen = (float2(1.0) - screen) * float2(image_region.min)
                               + screen * float2(image_region.max);

        Ray ray;
        ray.ori = U * screen.x + V * screen.y + W;
        ray.dir = dir;
        ray.tmin = 0.f;
        ray.tmax = FLT_MAX;
        return ray;
      }
      float3 dir,pos,up;
      float3 U, V, W;
      box2f image_region;
    } asOrthoCam;
  };

  template <typename RNG>
  VSNRAY_FUNC
  inline Ray primary_ray(RNG &rng, float x, float y, float width, float height) const
  {
    Ray ray;
    if (type == Pinhole)
      ray = asPinholeCam.primary_ray(Ray{}, rng, x, y, width, height);
    else if (type == Ortho)
      ray = asOrthoCam.primary_ray(Ray{}, x, y, width, height);
    else if (type == Matrix)
      ray = asMatrixCam.primary_ray(Ray{}, x, y, width, height);

    ray.time = lerp(shutter.min, shutter.max, rng());

    return ray;
  }
};

inline Camera createCamera()
{
  Camera cam;
  memset(&cam,0,sizeof(cam));
  cam.type = Camera::Unknown;
  cam.camID = UINT_MAX;
  cam.shutter = {0.5f, 0.5f};
  return cam;
}

// Frame //

struct Frame
{
  unsigned frameID;
  unsigned frameCounter;
  uint2 size;
  float2 invSize;
  int perPixelBytes;
  bool stochasticRendering;

  anari::DataType colorType;
  anari::DataType depthType;
  anari::DataType normalType;
  anari::DataType albedoType;
  anari::DataType primIdType;
  anari::DataType objIdType;
  anari::DataType instIdType;

  uint8_t *pixelBuffer;
  float *depthBuffer;
  float3 *normalBuffer;
  float3 *albedoBuffer;
  float4 *motionVecBuffer;
  uint32_t *primIdBuffer;
  uint32_t *objIdBuffer;
  uint32_t *instIdBuffer;
  float4 *accumBuffer;

  struct {
    bool enabled;
    float alpha;
    float4 *currBuffer;
    float4 *prevBuffer;
    float3 *currAlbedoBuffer;
    float3 *prevAlbedoBuffer;
#ifdef WITH_CUDA
    cuda_texture_ref<float4, 2> history;
#elif defined(WITH_HIP)
    hip_texture_ref<float4, 2> history;
#else
    texture_ref<float4, 2> history;
#endif
  } taa;

  VSNRAY_FUNC
  inline PixelSample pixelSample(int x, int y) const
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
  inline PixelSample accumSample(int x, int y, int accumID, PixelSample s) const
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
  inline void toneMap(int x, int y, PixelSample s) const
  {
    const auto idx = y * size.x + x;
    auto *color = pixelBuffer + (idx * perPixelBytes);

    switch (colorType) {
    case ANARI_UFIXED8_VEC4: {
      auto c = cvt_uint32(s.color);
      memcpy(color, &c, sizeof(c));
      break;
    }
    case ANARI_UFIXED8_RGBA_SRGB: {
      auto c = cvt_uint32_srgb(s.color);
      memcpy(color, &c, sizeof(c));
      break;
    }
    case ANARI_FLOAT32_VEC4: {
      memcpy(color, &s.color, sizeof(s.color));
      break;
    }
    default:
      break;
    }
  }

  VSNRAY_FUNC
  inline void fillGBuffer(int x, int y, PixelSample s) const
  {
    const auto idx = y * size.x + x;

    if (motionVecBuffer)
      motionVecBuffer[idx] = s.motionVec;
    if (taa.currBuffer)
      taa.currBuffer[idx] = s.color;
    if (taa.currAlbedoBuffer)
      taa.currAlbedoBuffer[idx] = s.albedo;

    // for the remaining values, only update if
    // depth is closer than the previous sample
    if (!depthBuffer || s.depth > depthBuffer[idx])
      return;

    if (depthBuffer)
      depthBuffer[idx] = s.depth;
    if (normalBuffer)
      normalBuffer[idx] = s.Ng;
    if (albedoBuffer)
      albedoBuffer[idx] = s.albedo;
    if (primIdBuffer)
      primIdBuffer[idx] = s.primId;
    if (objIdBuffer)
      objIdBuffer[idx] = s.objId;
    if (instIdBuffer)
      instIdBuffer[idx] = s.instId;
  }

  VSNRAY_FUNC
  inline void writeSample(int x, int y, int accumID, PixelSample s) const
  {
    fillGBuffer(x, y, s);
    toneMap(x, y, accumSample(x, y, accumID, s));
  }
};

inline Frame createFrame()
{
  Frame frame;
  memset(&frame,0,sizeof(frame));
  frame.frameID = UINT_MAX;
  frame.perPixelBytes = 1;
  frame.colorType = ANARI_UNKNOWN;
  frame.depthType = ANARI_UNKNOWN;
  frame.normalType = ANARI_UNKNOWN;
  frame.albedoType = ANARI_UNKNOWN;
  frame.primIdType = ANARI_UNKNOWN;
  frame.objIdType = ANARI_UNKNOWN;
  frame.instIdType = ANARI_UNKNOWN;
  frame.taa.alpha = 0.3f;
  return frame;
}

} // namespace visionaray::dco
