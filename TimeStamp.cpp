// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// std
#include <atomic>
// ours
#include "TimeStamp.h"

namespace visionaray {

std::atomic<TimeStamp> g_timeStamp{0ull};

TimeStamp newTimeStamp() {
  return ++g_timeStamp;
}

} // namespace visionaray
