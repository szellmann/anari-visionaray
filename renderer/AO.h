
#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

VSNRAY_FUNC
inline bool occluded(ScreenSample &ss, const Ray &ray, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice)
{
  auto hr = intersectSurfaces<1>(ss, ray, onDevice, worldID);
  auto hrv = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice);
  return hr.hit || hrv.hit;
}

VSNRAY_FUNC
inline float computeAO(ScreenSample &ss, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice,
    vec3 Ng, vec3 Ns, const vec3 viewDir, const vec3 isectPos,
    float time, float eps, int AO_samples, float AO_radius)
{
  vec3 u, v, w = Ns;
  make_orthonormal_basis(u,v,w);
  float weights = 0.f;
  float aoCount = 0.f;
  for (int i=0; i<AO_samples; ++i) {
    auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
    vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

    Ray aoRay;
    aoRay.ori = isectPos + Ns * eps;
    aoRay.dir = dir;
    aoRay.tmin = 0.f;
    aoRay.tmax = AO_radius;
    aoRay.time = time;

    float weight = max(0.f, dot(dir,Ns));
    weights += weight;
    if (weight > 0.f && occluded(ss, aoRay, worldID, onDevice)) {
      aoCount += weight;
    }
  }

  return weights==0.f ? 1.f : aoCount/weights;
}

} // namespace visionaray
