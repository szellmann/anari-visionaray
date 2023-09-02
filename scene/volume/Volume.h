// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

VSNRAY_FUNC
inline float4 postClassify(dco::Volume vol, float v) {
  if (vol.type == dco::Volume::TransferFunction1D) {
    box1 valueRange = vol.asTransferFunction1D.valueRange;
    v = (v - valueRange.min) / (valueRange.max - valueRange.min);
    return tex1D(vol.asTransferFunction1D.transFuncSampler, v);
  }

  return {};
}

struct Volume : public Object
{
  Volume(VisionarayGlobalState *d);
  virtual ~Volume();
  static Volume *createInstance(std::string_view subtype, VisionarayGlobalState *d);
  virtual aabb bounds() const = 0;
  dco::Geometry visionarayGeometry() const;

 protected:

  dco::Geometry vgeom;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Volume *, ANARI_VOLUME);
