// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Perspective.h"

namespace visionaray {

Perspective::Perspective(VisionarayGlobalState *s) : Camera(s)
{
  vcam.type = dco::Camera::Pinhole;
}

void Perspective::commitParameters()
{
  Camera::commitParameters();

  // NOTE: demonstrate alternative 'raw' method for getting parameter values
  if (!getParam("fovy", ANARI_FLOAT32, &m_fovy))
    m_fovy = 60.f * constants::degrees_to_radians<float>();
  m_aspect = getParam<float>("aspect", 1.f);
  m_apertureRadius = getParam<float>("apertureRadius", 0.f);
  m_focusDistance = getParam<float>("focusDistance", 1.f);

  // float2 imgPlaneSize;
  // imgPlaneSize.y = 2.f * tanf(0.5f * fovy);
  // imgPlaneSize.x = imgPlaneSize.y * aspect;

  // m_dir_du = normalize(cross(m_dir, m_up)) * imgPlaneSize.x;
  // m_dir_dv = normalize(cross(m_dir_du, m_dir)) * imgPlaneSize.y;
  // m_dir_00 = m_dir - .5f * m_dir_du - .5f * m_dir_dv;
}

void Perspective::finalize()
{
  Camera::finalize();
  vcam.asPinholeCam.perspective(m_fovy, m_aspect, .001f, 1000.f);
  //vcam.asPinholeCam.look_at(m_pos, m_pos + m_dir, m_up);
  vcam.asPinholeCam.set_eye(m_pos);
  vcam.asPinholeCam.set_dir(m_dir);
  vcam.asPinholeCam.set_up(m_up);
  vcam.asPinholeCam.set_image_region(m_imageRegion);
  vcam.asPinholeCam.set_lens_radius(m_apertureRadius);
  vcam.asPinholeCam.set_focal_distance(m_focusDistance);
}

} // namespace visionaray
