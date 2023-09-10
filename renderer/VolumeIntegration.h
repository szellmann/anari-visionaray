#pragma once

#include "renderer/common.h"
#include "DeviceCopyableObjects.h"

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

VSNRAY_FUNC
inline float4 postClassify(ScreenSample &ss, dco::TransferFunction tf, float v) {
  if (tf.type == dco::TransferFunction::_1D) {
    box1 valueRange = tf.as1D.valueRange;
    v = (v - valueRange.min) / (valueRange.max - valueRange.min);
    float4 clr = tex1D(tf.as1D.sampler, v);
    // if (ss.debug()) {
    //   printf("v: %f, clr: (%f,%f,%f)\n",v,clr.x,clr.y,clr.z);
    // }
    return clr;
  }

  return {};
}

VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            VisionarayGlobalState::DeviceObjectRegistry onDevice,
                            float3 &color,
                            float &alpha) {
  float dt = onDevice.spatialFields[vol.fieldID].baseDT;
  auto boxHit = intersect(ray, vol.bounds);
  // if (ss.debug()) {
  //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
  //   print(ray);
  //   print(vol.bounds);
  // }
  float t=boxHit.tnear;
  for (;t<boxHit.tfar&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(onDevice.spatialFields[vol.fieldID],P,v)) {
      float4 sample
          = postClassify(ss,onDevice.transferFunctions[vol.volID],v);
      color += dt * (1.f-alpha) * sample.w * sample.xyz();
      alpha += dt * (1.f-alpha) * sample.w;
    }
  }
  return t;
}

} // namespace visionaray
