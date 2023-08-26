// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceCopyableObjects.h"
#include "Object.h"

namespace visionaray {

struct Camera : public Object
{
  Camera(VisionarayGlobalState *s);
  ~Camera() override;

  virtual void commit() override;

  static Camera *createInstance(
      std::string_view type, VisionarayGlobalState *state);

  dco::Camera visionarayCamera() const { return vcam; }

 protected:
  dco::Camera vcam;
  vec3f m_pos;
  vec3f m_dir;
  vec3f m_up;
  box2f m_imageRegion;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Camera *, ANARI_CAMERA);
