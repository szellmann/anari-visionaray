// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "DeviceBVH.h"
#include "Geometry.h"

namespace visionaray {

struct Sphere : public Geometry
{
  Sphere(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  DeviceBVH<basic_sphere<float>> m_BVH;

  HostDeviceArray<basic_sphere<float>> m_spheres;
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexRadius;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  float m_globalRadius{0.f};

  HostDeviceArray<uint3> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
