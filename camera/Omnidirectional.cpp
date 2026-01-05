// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Omnidirectional.h"

namespace visionaray {

Omnidirectional::Omnidirectional(VisionarayGlobalState *s) : Camera(s)
{
  vcam.type = dco::Camera::Omni;
}

void Omnidirectional::finalize()
{
  Camera::finalize();
  vcam.asOmniCam.init(m_pos, m_dir, m_up);
}

} // namespace visionaray
