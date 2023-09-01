
#pragma once

#include "array/Array1D.h"
#include "SpatialField.h"

namespace visionaray {

struct UnstructuredField : public SpatialField
{
  UnstructuredField(VisionarayGlobalState *d);

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;
 
 private:

  aligned_vector<float4> m_vertices;
  aligned_vector<dco::UElem> m_elements;
  index_bvh<dco::UElem> m_samplingBVH;

  struct Parameters
  {
    helium::IntrusivePtr<Array1D> vertexPosition;
    helium::IntrusivePtr<Array1D> vertexData;
    helium::IntrusivePtr<Array1D> index;
    helium::IntrusivePtr<Array1D> cellIndex;
  } m_params;
  anari::DataType m_type{ANARI_UNKNOWN};
};

} // namespace visionaray
