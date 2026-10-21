// Copyright 2023-2026 Stefan Zellmann
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
  m_near = getParam<float>("near", 0.f);
  m_far = getParam<float>("far", FLT_MAX);
  m_apertureRadius = getParam<float>("apertureRadius", 0.f);
  m_focusDistance = getParam<float>("focusDistance", 1.f);
}

void Perspective::finalize()
{
  Camera::finalize();

  if (m_near < 0.f) { // == 0.f means near is not set
    reportMessage(ANARI_SEVERITY_WARNING, "near must be > 0");
    return;
  }

  if (m_far <= m_near) { // == 0.f means near is not set
    reportMessage(ANARI_SEVERITY_WARNING, "near must be < far");
    return;
  }

  vcam.asPinholeCam.init(m_pos, m_dir, m_up, m_fovy, m_aspect, m_near, m_far,
      m_imageRegion, m_apertureRadius, m_focusDistance);
}

} // namespace visionaray
