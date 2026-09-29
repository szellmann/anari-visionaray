// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "array/Array1D.h"
#include "DeviceArray.h"
#include "DeviceBVH.h"
#include "Volume.h"

namespace visionaray {

struct GaussianSplat : public Volume
{
  GaussianSplat(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  aabb bounds() const override;

 private:

  DeviceBVH<dco::Gaussian> m_splatBVH;

  HostDeviceArray<dco::Gaussian> m_gaussians;
  HostDeviceArray<float3> m_colors;

  aabb m_bounds;

  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexScale;
  helium::ChangeObserverPtr<Array1D> m_vertexRotation;
  helium::ChangeObserverPtr<Array1D> m_vertexOpacity;
  helium::ChangeObserverPtr<Array1D> m_f_dc;

  HostDeviceArray<uint3> vindex;
};

} // namespace visionaray
