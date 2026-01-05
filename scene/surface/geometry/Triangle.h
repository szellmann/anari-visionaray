// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct Triangle : public Geometry
{
  Triangle(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexNormal;
  helium::ChangeObserverPtr<Array1D> m_vertexTangent;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;

  HostDeviceArray<basic_triangle<3, float>> m_triangles;
  HostDeviceArray<uint3> vindex;
  HostDeviceArray<float3> vnormals;
  HostDeviceArray<float4> vtangents;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
