// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../Object.h"

namespace visionaray {

struct Camera : public Object
{
  Camera(VisionarayGlobalState *s);
  ~Camera() override;

  virtual void commit() override;

  static Camera *createInstance(
      std::string_view type, VisionarayGlobalState *state);

  virtual visionaray::ray createRay(const vec2f &screen) const = 0;

 protected:
  visionaray::vec3f m_pos;
  visionaray::vec3f m_dir;
  visionaray::vec3f m_up;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Camera *, ANARI_CAMERA);
