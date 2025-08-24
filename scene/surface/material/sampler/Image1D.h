// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sampler.h"
#include "array/Array1D.h"

namespace visionaray {

struct Image1D : public Sampler
{
  Image1D(VisionarayGlobalState *d);

  bool isValid() const override;
  void commitParameters() override;
  void finalize() override;

 private:
  void updateImageData();

  helium::IntrusivePtr<Array1D> m_image;
  dco::Attribute m_inAttribute{dco::Attribute::None};
  tex_address_mode m_wrapMode{Clamp};
  bool m_linearFilter{true};
  mat4 m_inTransform{mat4::identity()};
  float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  mat4 m_outTransform{mat4::identity()};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

#ifdef WITH_CUDA
  cuda_texture<vector<4, unorm<8>>, 1> vimage;
#elif defined(WITH_HIP)
  hip_texture<vector<4, unorm<8>>, 1> vimage;
#else
  texture<vector<4, unorm<8>>, 1> vimage;
#endif
};

} // namespace visionaray
