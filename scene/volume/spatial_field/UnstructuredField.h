
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

  aligned_vector<float4> m_vertices;
  aligned_vector<dco::UElem> m_elements;
  // for stitcher
  aligned_vector<int3> m_gridDims;
  aligned_vector<aabb> m_gridDomains;
  aligned_vector<uint64_t> m_gridScalarsOffsets;
  aligned_vector<float> m_gridScalars;
  // sampling accel
  index_bvh<dco::UElem> m_samplingBVH;

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
