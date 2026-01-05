// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// nanovdb
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#ifdef WITH_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif
// ours
#include "SpatialField.h"
#include "array/Array1D.h"

namespace visionaray {

struct NanoVDBField : public SpatialField
{
  NanoVDBField(VisionarayGlobalState *d);
  ~NanoVDBField();

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;
 private:

  std::string m_filter;

  helium::IntrusivePtr<Array1D> m_gridData;

  HostDeviceArray<uint8_t> m_deviceGrid;

  aabb m_bounds;
};

} // namespace visionaray
