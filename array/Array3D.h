// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../VisionarayGlobalState.h"
// helium
#include "helium/array/Array3D.h"

namespace visionaray {

using Array3DMemoryDescriptor = helium::Array3DMemoryDescriptor;

struct Array3D : public helium::Array3D
{
  Array3D(VisionarayGlobalState *state, const Array3DMemoryDescriptor &d);
  ~Array3D() override;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Array3D *, ANARI_ARRAY3D);
