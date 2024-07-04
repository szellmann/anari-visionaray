// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/texture/texture.h"
// ours
#include "Volume.h"
#include "array/Array1D.h"
#include "spatial_field/SpatialField.h"

namespace visionaray {

struct TransferFunction1D : public Volume
{
  TransferFunction1D(VisionarayGlobalState *d);
  ~TransferFunction1D() override;

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;

 private:
  void dispatch();

  // Data //

  helium::ChangeObserverPtr<SpatialField> m_field;

  aabb m_bounds;

  box1 m_valueRange{0.f, 1.f};
  float m_unitDistance{1.f};

  helium::ChangeObserverPtr<Array1D> m_colorData;
  helium::ChangeObserverPtr<Array1D> m_opacityData;

#ifdef WITH_CUDA
  cuda_texture<float4, 1> transFuncTexture;
#elif defined(WITH_HIP)
  hip_texture<float4, 1> transFuncTexture;
#else
  texture<float4, 1> transFuncTexture;
#endif

  HostDeviceArray<dco::Volume> m_volume;
};

} // namespace visionaray
