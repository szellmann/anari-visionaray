// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Perspective.h"

namespace visionaray {

Perspective::Perspective(VisionarayGlobalState *s) : Camera(s) {}

void Perspective::commit()
{
  Camera::commit();

  // NOTE: demonstrate alternative 'raw' method for getting parameter values
  float fovy = 0.f;
  if (!getParam("fovy", ANARI_FLOAT32, &fovy))
    fovy = 60.f * constants::degrees_to_radians<float>();
  float aspect = getParam<float>("aspect", 1.f);
  float apertureRadius = getParam<float>("apertureRadius", 0.f);
  float focusDistance = getParam<float>("focusDistance", 1.f);

  // float2 imgPlaneSize;
  // imgPlaneSize.y = 2.f * tanf(0.5f * fovy);
  // imgPlaneSize.x = imgPlaneSize.y * aspect;

  // m_dir_du = normalize(cross(m_dir, m_up)) * imgPlaneSize.x;
  // m_dir_dv = normalize(cross(m_dir_du, m_dir)) * imgPlaneSize.y;
  // m_dir_00 = m_dir - .5f * m_dir_du - .5f * m_dir_dv;

  vcam.type = dco::Camera::Pinhole;
  vcam.asPinholeCam.perspective(fovy, aspect, .001f, 1000.f);
  vcam.asPinholeCam.look_at(m_pos, m_pos + m_dir, m_up);
  vcam.asPinholeCam.set_image_region(m_imageRegion);
  vcam.asPinholeCam.set_lens_radius(apertureRadius);
  vcam.asPinholeCam.set_focal_distance(1.f);//focusDistance);
}

} // namespace visionaray
