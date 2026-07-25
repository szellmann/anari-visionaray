// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "DeviceBVH.h"
#include "Geometry.h"

namespace visionaray {

struct Triangle : public Geometry
{
  Triangle(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  DeviceBVH<dco::Triangle> m_BVH;

  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexNormal, m_faceVaryingNormal;
  helium::ChangeObserverPtr<Array1D> m_vertexTangent, m_faceVaryingTangent;
  std::array<helium::IntrusivePtr<Array1D>, 5>
      m_vertexAttributes, m_faceVaryingAttributes;

  HostDeviceArray<dco::Triangle> m_triangles;
  HostDeviceArray<uint3> vindex;
  HostDeviceArray<float3> vnormals, fvnormals;
  HostDeviceArray<float4> vtangents, fvtangents;
  HostDeviceArray<uint8_t> vattributes[5], fvattributes[5];
};

} // namespace visionaray
