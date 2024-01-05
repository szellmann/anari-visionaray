
#pragma once

#include "SpatialField.h"
#include "array/Array1D.h"
#include "array/ObjectArray.h"

namespace visionaray {

struct BlockStructuredField : public SpatialField
{
  BlockStructuredField(VisionarayGlobalState *d);

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;

 private:

  aligned_vector<dco::Block> m_blocks;
  aligned_vector<float> m_scalars;
  // sampling accel
  index_bvh<dco::Block> m_samplingBVH;

  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> cellWidth;
    helium::IntrusivePtr<helium::Array1D> blockBounds;
    helium::IntrusivePtr<helium::Array1D> blockLevel;
    helium::IntrusivePtr<helium::ObjectArray> blockData;
  } m_params;
};

} // namespace visionaray
