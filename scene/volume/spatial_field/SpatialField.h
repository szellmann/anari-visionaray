// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace visionaray {

VSNRAY_FUNC
inline bool sampleField(dco::SpatialField sf, vec3 P, float &value) {
  if (sf.type == dco::SpatialField::StructuredRegular) {
    value = tex3D(sf.asStructuredRegular.sampler,
        sf.asStructuredRegular.objectToTexCoord(P));
    return true;
  } else if (sf.type == dco::SpatialField::Unstructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;
    auto hr = intersect(ray, sf.asUnstructured.samplingBVH);

    if (!hr.hit)
      return false;

    value = hr.u; // value is stored in "u"!
    return true;
  }

  return false;
}

struct SpatialField : public Object
{
  SpatialField(VisionarayGlobalState *d);
  virtual ~SpatialField();
  static SpatialField *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  virtual aabb bounds() const = 0;

  float stepSize() const;

 protected:
  dco::SpatialField vfield;

  void setStepSize(float size);
  void dispatch();
  void detach();
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(
    visionaray::SpatialField *, ANARI_SPATIAL_FIELD);
