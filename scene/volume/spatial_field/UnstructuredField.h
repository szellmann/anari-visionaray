
#pragma once

#include "array/Array1D.h"
#include "array/ObjectArray.h"
#include "SpatialField.h"

namespace visionaray {

struct UnstructuredField : public SpatialField
{
  UnstructuredField(VisionarayGlobalState *d);

  void commit() override;

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
  index_bvh<dco::UElem> m_samplingBVH;
#endif

  struct Parameters
  {
    helium::IntrusivePtr<Array1D> vertexPosition;
    helium::IntrusivePtr<Array1D> vertexData;
    helium::IntrusivePtr<Array1D> index;
    helium::IntrusivePtr<Array1D> cellIndex;
    // "stitcher" extensions
    helium::IntrusivePtr<ObjectArray> gridData;
    helium::IntrusivePtr<Array1D> gridDomains;
  } m_params;
  anari::DataType m_type{ANARI_UNKNOWN};
};

} // namespace visionaray
