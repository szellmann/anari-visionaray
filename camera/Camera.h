// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../Object.h"
// visionaray
#include "visionaray/pinhole_camera.h"
#include "visionaray/matrix_camera.h"

namespace visionaray {

struct VisionarayCamera
{
  enum Type { Matrix, Pinhole, };
  Type type;
  union {
    matrix_camera asMatrixCam;
    pinhole_camera asPinholeCam;
  };
};

struct Camera : public Object
{
  Camera(VisionarayGlobalState *s);
  ~Camera() override;

  virtual void commit() override;

  static Camera *createInstance(
      std::string_view type, VisionarayGlobalState *state);

  VisionarayCamera getInternalCamera() const { return vcam; }

 protected:
  VisionarayCamera vcam;
  vec3f m_pos;
  vec3f m_dir;
  vec3f m_up;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Camera *, ANARI_CAMERA);
