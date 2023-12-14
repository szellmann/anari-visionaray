// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../VisionarayGlobalState.h"
#include "Array1D.h"
// helium
#include "helium/array/ObjectArray.h"

namespace visionaray {

struct ObjectArray : public helium::ObjectArray
{
  ObjectArray(VisionarayGlobalState *state, const Array1DMemoryDescriptor &d);
  ~ObjectArray() override;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::ObjectArray *, ANARI_ARRAY1D);
