// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../VisionarayGlobalState.h"
// helium
#include "helium/array/Array2D.h"

namespace visionaray {

using Array2DMemoryDescriptor = helium::Array2DMemoryDescriptor;

struct Array2D : public helium::Array2D
{
  Array2D(VisionarayGlobalState *state, const Array2DMemoryDescriptor &d);
  ~Array2D() override;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Array2D *, ANARI_ARRAY2D);
