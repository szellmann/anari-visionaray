// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "VisionarayGlobalState.h"

namespace visionaray {

VisionarayGlobalState::VisionarayGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), syncContext(std::make_shared<SyncContext>())
{}

VisionarayGlobalState::~VisionarayGlobalState()
{}

} // namespace visionaray
