// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dco/common.h"
#include "scene/volume/spatial_field/Plane.h"
#include "scene/volume/spatial_field/UElems.h"

#ifdef WITH_NANOVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>
#endif

namespace visionaray::dco {

// AMR dual mesh "gridlet" //

struct UElemGrid
{
  uint32_t gridID;
  int3 dims;
  aabb domain;
  uint64_t scalarsOffset;
  float *scalarsBuffer;
};

VSNRAY_FUNC
inline aabb get_bounds(const UElemGrid &grid)
{
  return grid.domain;
}

inline void split_primitive(
    aabb &L, aabb &R, float plane, int axis, const UElemGrid &grid)
{
  assert(0);
}

VSNRAY_FUNC
inline bool intersectGrid(const UElemGrid &grid, const float3 P, float& retVal)
{
  if (!grid.domain.contains(P))
    return false;

  int3 dims = grid.dims;
  int3 numScalars = dims+int3(1);
  float3 cellSize = grid.domain.size()/float3(dims);
  float3 objPos = (P-grid.domain.min)/cellSize;
  int3 imin(objPos);
  int3 imax = min(imin+int3(1),numScalars-int3(1));

  auto linearIndex = [numScalars](const int x, const int y, const int z) {
                       return z*numScalars.y*numScalars.x + y*numScalars.x + x;
                     };

  const float *scalars = grid.scalarsBuffer + grid.scalarsOffset;

  float f1 = scalars[linearIndex(imin.x,imin.y,imin.z)];
  float f2 = scalars[linearIndex(imax.x,imin.y,imin.z)];
  float f3 = scalars[linearIndex(imin.x,imax.y,imin.z)];
  float f4 = scalars[linearIndex(imax.x,imax.y,imin.z)];

  float f5 = scalars[linearIndex(imin.x,imin.y,imax.z)];
  float f6 = scalars[linearIndex(imax.x,imin.y,imax.z)];
  float f7 = scalars[linearIndex(imin.x,imax.y,imax.z)];
  float f8 = scalars[linearIndex(imax.x,imax.y,imax.z)];

#define EMPTY(x) isnan(x)
  if (EMPTY(f1) || EMPTY(f2) || EMPTY(f3) || EMPTY(f4) ||
      EMPTY(f5) || EMPTY(f6) || EMPTY(f7) || EMPTY(f8))
    return false;

  float3 frac = objPos-float3(imin);

  float f12 = lerp_r(f1,f2,frac.x);
  float f56 = lerp_r(f5,f6,frac.x);
  float f34 = lerp_r(f3,f4,frac.x);
  float f78 = lerp_r(f7,f8,frac.x);

  float f1234 = lerp_r(f12,f34,frac.y);
  float f5678 = lerp_r(f56,f78,frac.y);

  retVal = lerp_r(f1234,f5678,frac.z);

  return true;
}

template <typename R>
VSNRAY_FUNC
inline hit_record<R, primitive<unsigned>> intersect(const R &ray, const UElemGrid &grid)
{
  hit_record<R, primitive<unsigned>> result;
  float3 pos = ray.ori;
  float value = 0.f;

  bool hit = intersectGrid(grid, pos, value);
  result.hit = hit;

  if (result.hit) {
    result.t = 0.f;
    result.prim_id = grid.gridID;
    result.u = value; // misuse "u" to store value
  }

  return result;
}

// Unstructured element primitive //

struct UElem
{
  enum Type { Tet, Pyr, Wedge, Hex, BezierHex, Unknown, };
  Type type;
  uint64_t begin;
  uint64_t end;
  uint64_t elemID;
  // vertex data takes precedence; this value is used
  // if vertex value has special value NAN!
  float cellValue;
  const uint64_t *indexBuffer;
  const float4 *vertexBuffer;
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
  hit_record<Ray, primitive<unsigned>> result;
  float3 pos = ray.ori;
  float value = 0.f;

  uint64_t numVerts = elem.end-elem.begin;

  assert(numVerts>=4 && numVerts<=8);

  float4 v[8];
  for (int i=0; i<numVerts; ++i) {
    uint64_t idx = elem.indexBuffer[elem.begin+i];
    v[i] = elem.vertexBuffer[idx];
  }

  bool hit=numVerts==4 && intersectTet(value,pos,v[0],v[1],v[2],v[3])
        || numVerts==5 && intersectPyrEXT(value,pos,v[0],v[1],v[2],v[3],v[4])
        || numVerts==6 && intersectWedgeEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5])
        || numVerts==8 && intersectHexEXT(value,pos,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);

  // no vertex data: use cell data instead
  if (isnan(v[0].w)) {
    value = elem.cellValue;
  }

  result.hit = hit;

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
  box3 gridBounds; // in voxel/grid space
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

VSNRAY_FUNC
inline GridAccel createGridAccel()
{
  GridAccel accel;
  memset(&accel,0,sizeof(accel));
  accel.dims = int3(0);
  accel.worldBounds = box3f(float3(FLT_MAX),float3(-FLT_MAX));
  accel.gridBounds = box3f(float3(FLT_MAX),float3(-FLT_MAX));
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
  float cellSize;
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
      // Sampling BVHs, in _voxel_ space (make sure to xform ray first):
#ifdef WITH_CUDA
      cuda_bvh<UElem>::bvh_ref elemBVH;
      cuda_bvh<UElemGrid>::bvh_ref gridBVH;
#elif defined(WITH_HIP)
      hip_bvh<UElem>::bvh_ref elemBVH;
      hip_bvh<UElemGrid>::bvh_ref gridBVH;
#else
      bvh4<UElem>::bvh_ref elemBVH;
      bvh4<UElemGrid>::bvh_ref gridBVH;
#endif
      // for marcher:
#ifdef WITH_CUDA
      cuda_bvh<basic_triangle<3,float>>::bvh_ref shellBVH;
#elif defined(WITH_HIP)
      hip_bvh<basic_triangle<3,float>>::bvh_ref shellBVH;
#else
      bvh4<basic_triangle<3,float>>::bvh_ref shellBVH;
#endif
      const UElem *elems;
      const uint64_t *faceNeighbors;
      const basic_triangle<3,float> *shell;
    } asUnstructured;
    struct {
#ifdef WITH_CUDA
      cuda_bvh<Block>::bvh_ref samplingBVH;
#elif defined(WITH_HIP)
      hip_bvh<Block>::bvh_ref samplingBVH;
#else
      bvh4<Block>::bvh_ref samplingBVH;
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

VSNRAY_FUNC
inline SpatialField createSpatialField()
{
  SpatialField field;
  memset(&field,0,sizeof(field));
  field.type = SpatialField::Unknown;
  field.fieldID = UINT_MAX;
  field.cellSize = 1.0f;
  return field;
}

VSNRAY_FUNC
#ifdef __CUDACC__
// This fixes a crash with CUDA 13.3 and driver 610.57.04 where the
// kernel reports "illegal memory access" even though this function
// never even gets executed:
static __noinline__
bool sampleField(const SpatialField &sf, vec3 P, float &value, int &primID) {
#else
inline bool sampleField(const SpatialField &sf, vec3 P, float &value, int &primID) {
#endif
  // This assumes that P is in voxel space!
  if (sf.type == SpatialField::StructuredRegular) {
    value = tex3D(sf.asStructuredRegular.sampler,P);
    primID = 0;
    return true;
  } else if (sf.type == SpatialField::Unstructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;
    default_intersector isect;

    if (sf.asUnstructured.elemBVH.num_nodes()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
      auto hr = intersect_rayN_bvh2<detail::AnyHit>(ray,
                                                    sf.asUnstructured.elemBVH,
                                                    isect);
  
#else
      auto hr = intersect_ray1_bvhN<detail::AnyHit>(ray,
                                                    sf.asUnstructured.elemBVH,
                                                    isect);
  
#endif
      if (hr.hit) {
        value = hr.u; // value is stored in "u"!
        primID = hr.prim_id;
        return true;
      }
    }

    if (sf.asUnstructured.gridBVH.num_nodes()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
      auto hr = intersect_rayN_bvh2<detail::AnyHit>(ray,
                                                    sf.asUnstructured.gridBVH,
                                                    isect);
  
#else
      auto hr = intersect_ray1_bvhN<detail::AnyHit>(ray,
                                                    sf.asUnstructured.gridBVH,
                                                    isect);
  
#endif
      if (hr.hit) {
        value = hr.u; // value is stored in "u"!
        primID = hr.prim_id;
        return true;
      }
    }

    return false;
  } else if (sf.type == SpatialField::BlockStructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;

    // sumValues+sumWeightedValues
    float basisPRD[2] = {0.f,0.f};
    ray.prd = &basisPRD;

    default_intersector isect;
#if defined(WITH_CUDA) || defined(WITH_HIP)
    auto hr = intersect(ray, sf.asBlockStructured.samplingBVH);
#else
    auto hr = intersect_ray1_bvhN<detail::AnyHit>(ray,
                                                  sf.asBlockStructured.samplingBVH,
                                                  isect);
#endif

    if (basisPRD[1] == 0.f)
      return false;

    value = basisPRD[0]/basisPRD[1];
    primID = hr.prim_id;
    return true;
  }
#ifdef WITH_NANOVDB
  else if (sf.type == SpatialField::NanoVDB) {
    auto acc = sf.asNanoVDB.grid->getAccessor();
    nanovdb::math::Vec3<float> nvdbPos(P.x,P.y,P.z);
    if (sf.asNanoVDB.filterMode == Nearest) {
      auto smp = nanovdb::math::createSampler<0>(acc);
      value = smp(sf.asNanoVDB.grid->worldToIndexF(nvdbPos));
      primID = 0;
      return true;
    } else if (sf.asNanoVDB.filterMode == Linear) {
      auto smp = nanovdb::math::createSampler<1>(acc);
      value = smp(sf.asNanoVDB.grid->worldToIndexF(nvdbPos));
      primID = 0;
      return true;
    }
  }
#endif

  return false;
}

VSNRAY_FUNC
inline bool sampleField(const SpatialField &sf, vec3 P, float &value) {
  int ignore;
  return sampleField(sf,P,value,ignore);
}

VSNRAY_FUNC
inline bool sampleGradient(const SpatialField &sf, vec3 P, vec3 delta, float3 &value) {
  float x0=0, x1=0, y0=0, y1=0, z0=0, z1=0;
  bool b0 = sampleField(sf, P+float3{delta.x, 0.f, 0.f}, x1);
  bool b1 = sampleField(sf, P-float3{delta.x, 0.f, 0.f}, x0);
  bool b2 = sampleField(sf, P+float3{0.f, delta.y, 0.f}, y1);
  bool b3 = sampleField(sf, P-float3{0.f, delta.y, 0.f}, y0);
  bool b4 = sampleField(sf, P+float3{0.f, 0.f, delta.z}, z1);
  bool b5 = sampleField(sf, P-float3{0.f, 0.f, delta.z}, z0);
  if (b0 && b1 && b2 && b3 && b4 && b5) {
    value = float3{x1,y1,z1}-float3{x0,y0,z0};
    return true; // TODO
  } else {
    value = float3{0.f};
    return false;
  }
}

} // namespace visionaray::dco
