// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "common.h"

namespace visionaray {

struct VisionaraySceneImpl;

struct VisionaraySceneGPU
{
  VisionaraySceneGPU(VisionaraySceneImpl *cpuScene);
  ~VisionaraySceneGPU();

  bool isValid() const;
  aabb getBounds() const;
  void commit();
  void dispatch();
  void attachInstance(dco::Instance inst, unsigned instID, unsigned userID=~0u);
  void attachGeometry(dco::Geometry geom, unsigned geomID, unsigned userID=~0u);

#ifdef WITH_CUDA
  cuda_index_bvh<dco::BLS>::bvh_ref refBVH();
#elif defined(WITH_HIP)
  hip_index_bvh<dco::BLS>::bvh_ref refBVH();
#elif defined(WITH_SYCL)
  sycl_index_bvh<dco::BLS>::bvh_ref refBVH();
#endif
 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  VisionarayGlobalState *deviceState();
};

} // namespace visionaray
