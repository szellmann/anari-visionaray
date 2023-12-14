// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../VisionarayGlobalState.h"
// helium
#include "helium/array/Array1D.h"

namespace visionaray {

using Array1DMemoryDescriptor = helium::Array1DMemoryDescriptor;

struct Array1D : public helium::Array1D
{
  Array1D(VisionarayGlobalState *state, const Array1DMemoryDescriptor &d);
  ~Array1D() override;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Array1D *, ANARI_ARRAY1D);
