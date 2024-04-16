// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct Cylinder : public Geometry
{
  Cylinder(VisionarayGlobalState *s);

  void commit() override;

  //float4 getAttributeValue(
  //    const Attribute &attr, const Ray &ray) const override;

 private:

  HostDeviceArray<basic_cylinder<float>> m_cylinders;
  helium::CommitObserverPtr<Array1D> m_index;
  helium::CommitObserverPtr<Array1D> m_radius;
  helium::CommitObserverPtr<Array1D> m_vertexPosition;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  float m_globalRadius{0.f};

  HostDeviceArray<uint2> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
