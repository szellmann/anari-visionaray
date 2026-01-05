// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct BezierCurve : public Geometry
{
  BezierCurve(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  HostDeviceArray<dco::BezierCurve> m_curves;
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_radius;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  float m_globalRadius{1.f};

  HostDeviceArray<uint3> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
