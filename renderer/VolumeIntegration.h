#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            float3 &color,
                            float &alpha) {
  float dt = vol.field.baseDT;
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
    if (sampleField(vol.field,P,v)) {
      float4 sample
          = postClassify(vol.asTransferFunction1D,v);
      color += dt * (1.f-alpha) * sample.w * sample.xyz();
      alpha += dt * (1.f-alpha) * sample.w;
    }
  }
  return t;
}

VSNRAY_FUNC
inline dco::HitRecordVolume sampleFreeFlightDistanceAllVolumes(
    ScreenSample &ss, Ray ray, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice) {

  ray.prd = &ss.random;
  return intersectVolumes(ray, onDevice.TLSs[worldID]);
}

} // namespace visionaray
