// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/Array3D.h"

namespace visionaray {

Array3D::Array3D(VisionarayGlobalState *state, const Array3DMemoryDescriptor &d)
    : helium::Array3D(state, d)
{
  state->objectCounts.arrays++;
}

Array3D::~Array3D()
{
  asVisionarayState(deviceState())->objectCounts.arrays--;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Array3D *);
