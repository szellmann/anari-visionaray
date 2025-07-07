// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Instance.h"
#include "VisionarayScene.h"

namespace visionaray {

struct World : public Object
{
  World(VisionarayGlobalState *s);
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  const std::vector<Instance *> &instances() const;

  VisionarayScene visionarayScene() const;
  void visionaraySceneUpdate();

 private:
  void rebuildBLSs();
  void recommitBLSs();
  void rebuildTLS();
  void cleanup();

  helium::ChangeObserverPtr<ObjectArray> m_zeroSurfaceData;
  helium::ChangeObserverPtr<ObjectArray> m_zeroVolumeData;
  helium::ChangeObserverPtr<ObjectArray> m_zeroLightData;

  helium::ChangeObserverPtr<ObjectArray> m_instanceData;
  std::vector<Instance *> m_instances;

  helium::IntrusivePtr<Group> m_zeroGroup;
  helium::IntrusivePtr<Instance> m_zeroInstance;

  size_t m_numSurfaceInstances{0};
  size_t m_numVolumeInstances{0};
  size_t m_numLightInstances{0};

  aabb m_surfaceBounds;

  struct ObjectUpdates
  {
    helium::TimeStamp lastTLSBuild{0};
    helium::TimeStamp lastBLSReconstructCheck{0};
    helium::TimeStamp lastBLSCommitCheck{0};
  } m_objectUpdates;
//
  VisionarayScene vscene{nullptr};
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::World *, ANARI_WORLD);
