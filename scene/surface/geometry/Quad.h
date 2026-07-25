// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "DeviceBVH.h"
#include "Geometry.h"

namespace visionaray {

struct Quad : public Geometry
{
  Quad(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  DeviceBVH<basic_triangle<3,float>> m_BVH;

  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexNormal, m_faceVaryingNormal;
  helium::ChangeObserverPtr<Array1D> m_vertexTangent, m_faceVaryingTangent;
  std::array<helium::IntrusivePtr<Array1D>, 5>
      m_vertexAttributes, m_faceVaryingAttributes;

  HostDeviceArray<basic_triangle<3, float>> m_triangles;
  HostDeviceArray<uint4> vindex;
  HostDeviceArray<float3> vnormals, fvnormals;
  HostDeviceArray<float4> vtangents, fvtangents;
  HostDeviceArray<uint8_t> vattributes[5], fvattributes[5];
};

} // namespace visionaray
