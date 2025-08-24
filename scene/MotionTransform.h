// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Instance.h"
#include "array/Array1D.h"

namespace visionaray {

struct MotionTransform : public Instance
{
  MotionTransform(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  void visionarayInstanceUpdate() override;

 private:
  helium::ChangeObserverPtr<Array1D> m_motionTransform;
  box1 m_time{0.f, 1.f};
};

} // namespace visionaray
