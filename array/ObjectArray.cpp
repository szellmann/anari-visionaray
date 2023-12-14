// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/ObjectArray.h"
#include "../VisionarayGlobalState.h"

namespace visionaray {

ObjectArray::ObjectArray(
    VisionarayGlobalState *state, const Array1DMemoryDescriptor &d)
    : helium::ObjectArray(state, d)
{
  state->objectCounts.arrays++;
}

ObjectArray::~ObjectArray()
{
  asVisionarayState(deviceState())->objectCounts.arrays--;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::ObjectArray *);
