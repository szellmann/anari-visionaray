// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/aligned_vector.h"
// ours
#include "Geometry.h"

namespace visionaray {

struct Sphere : public Geometry
{
  Sphere(VisionarayGlobalState *s);

  void commit() override;

  //float4 getAttributeValue(
  //    const Attribute &attr, const Ray &ray) const override;

 private:
  void cleanup();

  aligned_vector<basic_sphere<float>> m_spheres;
  helium::IntrusivePtr<Array1D> m_index;
  helium::IntrusivePtr<Array1D> m_vertexPosition;
  helium::IntrusivePtr<Array1D> m_vertexRadius;
  //std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  //std::vector<uint32_t> m_attributeIndex;
  float m_globalRadius{0.f};
};

} // namespace visionaray
