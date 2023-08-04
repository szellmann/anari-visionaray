// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#include "Instance.h"
#include "Object.h"

namespace visionaray {

struct World : public Object
{
  World(VisionarayGlobalState *s);
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

//  const std::vector<Instance *> &instances() const;
//
//  void intersectVolumes(VolumeRay &ray) const;
//
//  RTCScene embreeScene() const;
//  void embreeSceneUpdate();
//
// private:
//  void rebuildBLSs();
//  void recommitBLSs();
//  void rebuildTLS();
//  void cleanup();
//
//  helium::IntrusivePtr<ObjectArray> m_zeroSurfaceData;
//  helium::IntrusivePtr<ObjectArray> m_zeroVolumeData;
//
//  helium::IntrusivePtr<ObjectArray> m_instanceData;
//  std::vector<Instance *> m_instances;
//
//  bool m_addZeroInstance{false};
//  helium::IntrusivePtr<Group> m_zeroGroup;
//  helium::IntrusivePtr<Instance> m_zeroInstance;
//
//  size_t m_numSurfaceInstances{0};
//
//  visionaray::aabb m_surfaceBounds;
//
//  struct ObjectUpdates
//  {
//    helium::TimeStamp lastTLSBuild{0};
//    helium::TimeStamp lastBLSReconstructCheck{0};
//    helium::TimeStamp lastBLSCommitCheck{0};
//  } m_objectUpdates;
//
//  RTCScene m_embreeScene{nullptr};
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::World *, ANARI_WORLD);
