#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

template <bool Shading>
VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            const DeviceObjectRegistry &onDevice,
                            Ray ray,
                            const dco::Volume &vol,
                            const dco::LightRef *allLights,
                            unsigned numLights,
                            float3 ambientColor,
                            float ambientRadiance,
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

  float3 delta(sf.cellSize, sf.cellSize, sf.cellSize);
  delta *= float3(sf.voxelSpaceTransform(0,0),
                  sf.voxelSpaceTransform(1,1),
                  sf.voxelSpaceTransform(2,2));

  float3 viewDir = -ray.dir;

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

      // Gradient shading:
      float3 shadedColor = sample.xyz();
      if constexpr (Shading) {
        float3 gn(0.f);
        if (sampleGradient(sf,P,delta,gn))
          gn = normalize(gn);

        gn = faceforward(gn, viewDir, gn);

        if (length(gn) > 1e-10f) {
          auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
          for (unsigned lightID=0; lightID<numLights; ++lightID) {
            const dco::Light &light = getLight(allLights, lightID, onDevice);

            LightSample ls = sampleLight(light, P, ss.random);

            dco::Material mat = dco::createMaterial();
            mat.type = dco::Material::Matte;
            mat.asMatte.color = dco::createMaterialParamRGB();
            mat.asMatte.color.rgb = sample.xyz();

            shadedColor = evalMaterial(mat,
                                       onDevice,
                                       nullptr, // attribs, not used..
                                       float3(0.f), // objPos, not used..
                                       UINT_MAX, // primID, not used..
                                       gn, gn,
                                       normalize(viewDir),
                                       normalize(ls.dir),
                                       ls.intensity * safe_rcp(ls.dist2));
            shadedColor = shadedColor * safe_rcp(ls.pdf);
            shadedColor += sample.xyz() * ambientColor * ambientRadiance;
          }
        }
      }

      color += transmittance * (1.f - stepTransmittance) * shadedColor;
      alpha += transmittance * (1.f - stepTransmittance);
      transmittance *= stepTransmittance;
    }
  }
  return t;
}

} // namespace visionaray
