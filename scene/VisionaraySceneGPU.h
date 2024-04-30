
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
  void attachGeometry(dco::Geometry geom, unsigned geomID, unsigned userID=~0u);

  cuda_index_bvh<dco::BLS>::bvh_inst instBVH(mat4x3 xfm);
 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  VisionarayGlobalState *deviceState();
};

} // namespace visionaray
