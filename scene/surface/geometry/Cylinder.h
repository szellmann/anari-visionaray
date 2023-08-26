// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/aligned_vector.h"
// ours
#include "Geometry.h"

namespace visionaray {

struct Cylinder : public Geometry
{
  Cylinder(VisionarayGlobalState *s);

  void commit() override;

  //float4 getAttributeValue(
  //    const Attribute &attr, const Ray &ray) const override;

 private:
  void cleanup();

  aligned_vector<basic_cylinder<float>> m_cylinders;
  helium::IntrusivePtr<Array1D> m_index;
  helium::IntrusivePtr<Array1D> m_radius;
  helium::IntrusivePtr<Array1D> m_vertexPosition;
  //std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  float m_globalRadius{0.f};
};

} // namespace visionaray
