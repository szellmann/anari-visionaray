// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/Array1D.h"

namespace visionaray {

Array1D::Array1D(VisionarayGlobalState *state, const Array1DMemoryDescriptor &d)
    : helium::Array1D(state, d)
{
  state->objectCounts.arrays++;
}

Array1D::~Array1D()
{
  asVisionarayState(deviceState())->objectCounts.arrays--;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Array1D *);
