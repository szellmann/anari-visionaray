
#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

VSNRAY_FUNC
inline bool occluded(ScreenSample &ss, const Ray &ray, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice)
{
  auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
  auto hrv = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice);
  return hr.hit || hrv.hit;
}

VSNRAY_FUNC
inline float computeAO(ScreenSample &ss, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice,
    vec3 Ng, const vec3 viewDir, const vec3 isectPos, int AO_samples, float AO_radius)
{
  Ng = faceforward(Ng, viewDir, Ng);
  vec3 u, v, w = Ng;
  make_orthonormal_basis(u,v,w);
  float weights = 0.f;
  float aoCount = 0.f;
  for (int i=0; i<AO_samples; ++i) {
    auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
    vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

    Ray aoRay;
    aoRay.ori = isectPos + Ng * 1e-4f;
    aoRay.dir = dir;
    aoRay.tmin = 0.f;
    aoRay.tmax = AO_radius;

    float weight = max(0.f, dot(dir,Ng));
    weights += weight;
    if (weight > 0.f && occluded(ss, aoRay, worldID, onDevice)) {
      aoCount += weight;
    }
  }

  return weights==0.f ? 1.f : aoCount/weights;
}

} // namespace visionaray
