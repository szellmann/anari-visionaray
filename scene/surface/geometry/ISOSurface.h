// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "Geometry.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"

namespace visionaray {

struct ISOSurface : public Geometry
{
  ISOSurface(VisionarayGlobalState *d);
  ~ISOSurface() override;

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

 private:

  // Data //

  helium::ChangeObserverPtr<SpatialField> m_field;
  helium::ChangeObserverPtr<Array1D> m_isoValue;

  HostDeviceArray<dco::ISOSurface> m_isoSurface;
};

} // namespace visionaray
