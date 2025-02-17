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
  float dt = vol.unitDistance;
  auto boxHit = intersect(ray, vol.bounds);

  const auto &sf = vol.field;

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  dt = dt * dt_scale;

  // if (ss.debug()) {
  //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
  //   print(ray);
  //   print(vol.bounds);
  // }
  float t=ray.tmin;
  for (;t<ray.tmax&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(sf,P,v)) {
      float4 sample
          = postClassify(vol.asTransferFunction1D,v);
      float scale = 1.0f - powf(1.0f - sample.w, 1.f/dt);
      color += scale * (1.f-alpha) * sample.w * sample.xyz();
      alpha += scale * (1.f-alpha) * sample.w;
    }
  }
  return t / dt_scale;
}

} // namespace visionaray
