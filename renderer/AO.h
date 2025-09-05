// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

VSNRAY_FUNC
inline bool occluded(ScreenSample &ss, Ray ray, unsigned worldID,
    DeviceObjectRegistry onDevice, const RendererState &rendererState)
{
  ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);
  auto hr = intersectAll(ss, ray, worldID, onDevice, /*shadow:*/true);
  return hr.hit;
}

VSNRAY_FUNC
inline float computeAO(ScreenSample &ss, unsigned worldID,
    DeviceObjectRegistry onDevice, const RendererState &rendererState,
    vec3 Ng, vec3 Ns, const vec3 viewDir, const vec3 isectPos,
    float time, float eps)
{
  const int AO_samples = rendererState.ambientSamples;
  const float AO_radius = rendererState.occlusionDistance;

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
    if (weight > 0.f && occluded(ss, aoRay, worldID, onDevice, rendererState)) {
      aoCount += weight;
    }
  }

  return weights==0.f ? 1.f : aoCount/weights;
}

} // namespace visionaray
