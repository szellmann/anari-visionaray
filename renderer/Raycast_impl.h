#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

struct VisionarayRendererRaycast
{
  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts /*objCounts*/) {

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1e31f;

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
      const auto &mat = onDevice.groups[inst.groupID].materials[hr.geom_id];
      float3 shadedColor = to_rgb(mat.asMatte.data.shade(sr));

      result.color = float4(float3(.8f)*dot(-ray.dir,gn),1.f);
      result.color = float4(shadedColor,1.f);
      result.depth = hr.t;

      ray.tmax = hr.t;
    }

    hr = intersectVolumes(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      const auto &vol = geom.asVolume.data;
      float3 color(0.f);
      float alpha = 0.f;

      rayMarchVolume(ss, ray, vol, onDevice, color, alpha);
      result.color = over(float4(color,alpha), result.color);
    }

    if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
    }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
