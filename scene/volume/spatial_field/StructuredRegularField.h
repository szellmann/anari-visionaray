// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/texture/texture.h"
// ours
#include "SpatialField.h"
#include "array/Array3D.h"

namespace visionaray {

struct StructuredRegularField : public SpatialField
{
  StructuredRegularField(VisionarayGlobalState *d);

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;

 private:

  // Data //

  uint3 m_dims{0u};
  float3 m_origin;
  float3 m_spacing;
  std::string m_filter;

  helium::IntrusivePtr<Array3D> m_dataArray;

#ifdef WITH_CUDA
  cuda_texture<float, 3> m_dataTexture;
#elif defined(WITH_HIP)
  hip_texture<float, 3> m_dataTexture;
#else
  texture<float, 3> m_dataTexture;
#endif
  anari::DataType m_type{ANARI_UNKNOWN};
};

} // namespace visionaray
