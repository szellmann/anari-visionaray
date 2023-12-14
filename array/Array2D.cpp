// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/Array2D.h"
#include "../VisionarayGlobalState.h"

namespace visionaray {

Array2D::Array2D(VisionarayGlobalState *state, const Array2DMemoryDescriptor &d)
    : helium::Array2D(state, d)
{
  state->objectCounts.arrays++;
}

Array2D::~Array2D()
{
  asVisionarayState(deviceState())->objectCounts.arrays--;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Array2D *);
