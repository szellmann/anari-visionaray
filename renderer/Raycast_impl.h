#pragma once

#include "renderer/common.h"

namespace visionaray {

struct VisionarayRendererRaycast
{
  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts /*objCounts*/) {

    auto debug = [=]() {
      return prd.x == prd.frameSize.x/2 && prd.y == prd.frameSize.y/2;
    };

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1.f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      vec3f hitPos = ray.ori + hr.t * ray.dir;
      vec3f gn = getNormal(geom, hr.prim_id, hitPos);

      shade_record<float> sr;
      sr.normal = gn;
      sr.geometric_normal = gn;
      sr.view_dir = -ray.dir;
      sr.tex_color = float3(1.f);
      sr.light_dir = -ray.dir;
      sr.light_intensity = float3(1.f);

      // That doesn't work for instances..
      const auto &mat = onDevice.materials[hr.geom_id];
      float3 shadedColor = to_rgb(mat.asMatte.data.shade(sr));

      result.color = float4(float3(.8f)*dot(-ray.dir,gn),1.f);
      result.color = float4(shadedColor,1.f);

      ray.tmax = hr.t;
    }

    hr = intersectVolumes(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      const auto &vol = geom.asVolume.data;
      auto boxHit = intersect(ray, vol.bounds);
      float dt = onDevice.spatialFields[vol.fieldID].baseDT;
      float3 color(0.f);
      float alpha = 0.f;
      // if (debug()) {
      //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
      //   print(ray);
      //   print(vol.bounds);
      // }
      for (float t=boxHit.tnear;t<boxHit.tfar&&alpha<0.99f;t+=dt) {
        float3 P = ray.ori+ray.dir*t;
        float v = 0.f;
        if (sampleField(onDevice.spatialFields[vol.fieldID],P,v)) {
          float4 sample
              = postClassify(onDevice.transferFunctions[vol.volID],v);
          color += dt * (1.f-alpha) * sample.w * sample.xyz();
          alpha += dt * (1.f-alpha) * sample.w;
        }
      }

      result.color = over(float4(color,1.f), result.color);
    }

    if (prd.x == prd.frameSize.x/2 || prd.y == prd.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
    }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
