
#pragma once

#include "array/Array1D.h"
#include "array/ObjectArray.h"
#include "SpatialField.h"

namespace visionaray {

struct UnstructuredField : public SpatialField
{
  UnstructuredField(VisionarayGlobalState *d);

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;

 private:

  HostDeviceArray<float4> m_vertices;
  HostDeviceArray<uint64_t> m_indices;
  HostDeviceArray<dco::UElem> m_elements;
  // for stitcher
  HostDeviceArray<int3> m_gridDims;
  HostDeviceArray<aabb> m_gridDomains;
  HostDeviceArray<uint64_t> m_gridScalarsOffsets;
  HostDeviceArray<float> m_gridScalars;
  // sampling accel
#ifdef WITH_CUDA
  cuda_index_bvh<dco::UElem> m_samplingBVH;
#else
  bvh4<dco::UElem> m_samplingBVH;
#endif
  // shell accel
#ifdef WITH_CUDA
  //cuda_index_bvh<dco::UElem> m_samplingBVH;
#else
  // must be an *index* BVH so we can access the
  // triangles based on their prim_id in the shader:
  index_bvh4<basic_triangle<3,float>> m_shellBVH;
#endif
  // for marching
  HostDeviceArray<uint64_t> m_faceNeighbors;

  struct Parameters
  {
    helium::IntrusivePtr<Array1D> vertexPosition;
    helium::IntrusivePtr<Array1D> vertexData;
    helium::IntrusivePtr<Array1D> index;
    helium::IntrusivePtr<Array1D> cellIndex;
    helium::IntrusivePtr<Array1D> cellType;
    helium::IntrusivePtr<Array1D> cellData;
    // "stitcher" extensions
    helium::IntrusivePtr<ObjectArray> gridData;
    helium::IntrusivePtr<Array1D> gridDomains;
  } m_params;
  anari::DataType m_type{ANARI_UNKNOWN};
};

} // namespace visionaray
