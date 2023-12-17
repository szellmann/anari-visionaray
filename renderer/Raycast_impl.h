#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

struct VisionarayRendererRaycast
{
  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice) const {

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1e31f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      const auto &inst = onDevice.instances[hr.inst_id];
      const auto &group = onDevice.groups[inst.groupID];
      const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const auto &mat = onDevice.materials[group.materials[hr.geom_id]];

      vec3f hitPos = ray.ori + hr.t * ray.dir;
      vec3f gn = getNormal(geom, hr.prim_id, hitPos);
      vec2f uv{hr.u,hr.v};
      vec4f color{1.f};
      if (mat.type == dco::Material::Matte && mat.asMatte.samplerID < UINT_MAX) {
        const auto &samp = onDevice.samplers[mat.asMatte.samplerID];
        color = getSample(samp, geom, hr.prim_id, uv);
      } else {
        color = getColor(geom, mat, hr.prim_id, uv);
      }

      float3 xfmDir = (inst.invXfm * float4(ray.dir, 0.f)).xyz();

      shade_record<float> sr;
      sr.normal = gn;
      sr.geometric_normal = gn;
      sr.view_dir = -xfmDir;
      sr.tex_color = color.xyz();
      sr.light_dir = -xfmDir;
      sr.light_intensity = float3(1.f);

      // That doesn't work for instances..
      float3 shadedColor{0.f};

      if (rendererState.renderMode == RenderMode::Default)
        shadedColor = to_rgb(mat.asMatte.data.shade(sr));
      else if (rendererState.renderMode == RenderMode::Ng)
        shadedColor = (gn + float3(1.f)) * float3(0.5f);
      else if (rendererState.renderMode == RenderMode::Albedo)
        shadedColor = color.xyz();
      else if (rendererState.renderMode == RenderMode::GeometryAttribute0)
        shadedColor = getAttribute(geom, dco::Attribute::_0, hr.prim_id, uv).xyz();
      else if (rendererState.renderMode == RenderMode::GeometryAttribute1)
        shadedColor = getAttribute(geom, dco::Attribute::_1, hr.prim_id, uv).xyz();
      else if (rendererState.renderMode == RenderMode::GeometryAttribute2)
        shadedColor = getAttribute(geom, dco::Attribute::_2, hr.prim_id, uv).xyz();
      else if (rendererState.renderMode == RenderMode::GeometryAttribute3)
        shadedColor = getAttribute(geom, dco::Attribute::_3, hr.prim_id, uv).xyz();
      else if (rendererState.renderMode == RenderMode::GeometryColor)
        shadedColor = getAttribute(geom, dco::Attribute::Color, hr.prim_id, uv).xyz();

      result.color = float4(float3(.8f)*dot(-ray.dir,gn),1.f);
      result.color = float4(shadedColor,1.f);
      result.depth = hr.t;
      result.Ng = gn;
      result.Ns = gn;
      result.albedo = color.xyz();
      result.primId = hr.prim_id;
      result.objId = hr.geom_id;
      result.instId = hr.inst_id;

      ray.tmax = hr.t;
    }

    hr = intersectVolumes(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {
      const auto &inst = onDevice.instances[hr.inst_id];
      const auto &group = onDevice.groups[inst.groupID];
      const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const auto &vol = geom.asVolume.data;

      float3 color(0.f);
      float alpha = 0.f;

      rayMarchVolume(ss, ray, vol, onDevice, color, alpha);
      result.color = over(float4(color,alpha), result.color);
      result.Ng = float3{}; // TODO: gradient
      result.Ns = float3{}; // TODO..
      result.albedo = float3{}; // TODO..
      result.primId = hr.prim_id;
      result.objId = hr.geom_id;
      result.instId = hr.inst_id;
    }

    // if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
    //   result.color = float4(1.f) - result.color;
    // }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
