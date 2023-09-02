// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

VSNRAY_FUNC
inline float4 postClassify(dco::TransferFunction tf, float v, bool dbg=false) {
  if (tf.type == dco::TransferFunction::_1D) {
    box1 valueRange = tf.as1D.valueRange;
    v = (v - valueRange.min) / (valueRange.max - valueRange.min);
    float4 clr = tex1D(tf.as1D.sampler, v);
    if (dbg) {
      printf("v: %f, clr: (%f,%f,%f)\n",v,clr.x,clr.y,clr.z);
    }
    return clr;
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
