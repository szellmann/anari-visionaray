#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            float samplingRateInv,
                            float3 &color,
                            float &alpha) {
  auto boxHit = intersect(ray, vol.bounds);

  const auto &sf = vol.field;

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  float dt = sf.cellSize*samplingRateInv;

  // if (ss.debug()) {
  //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
  //   print(ray);
  //   print(vol.bounds);
  // }
  float t=ray.tmin;
  float transmittance = 1.f;
  for (;t<ray.tmax&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(sf,P,v)) {
      float4 sample
          = postClassify(vol.asTransferFunction1D,v);
      float stepTransmittance =
          powf(1.f - sample.w, dt / vol.unitDistance);
      color += transmittance * (1.f - stepTransmittance) * sample.xyz();
      alpha += transmittance * (1.f - stepTransmittance);
      transmittance *= stepTransmittance;
    }
  }
  return t;
}

} // namespace visionaray
