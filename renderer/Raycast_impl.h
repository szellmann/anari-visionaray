#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

struct VisionarayRendererRaycast
{
  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        const VisionarayGlobalState::DeviceObjectRegistry &onDevice) const {

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1e31f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);

    bool hit = false;

    float3 surfaceColor{0.f};
    float surfaceAlpha = 0.f;
    while (true) { // all transparent surfaces
      if (!hr.hit) break;
      const auto &inst = onDevice.instances[hr.inst_id];
      const auto &group = onDevice.groups[inst.groupID];
      const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const auto &mat = onDevice.materials[group.materials[hr.geom_id]];

      vec3f hitPos = ray.ori + hr.t * ray.dir;
      vec2f uv{hr.u,hr.v};
      vec3f gn = getNormal(geom, hr.prim_id, hitPos);
      vec3f sn = getShadingNormal(geom, hr.prim_id, hitPos, uv);
      vec3f tng{0.f};
      vec3f btng{0.f};
      float4 tng4 = getTangent(geom, hr.prim_id, hitPos, uv);
      if (length(sn) > 0.f && length(tng4.xyz()) > 0.f) {
        tng = tng4.xyz();
        btng = cross(sn, tng) * tng4.w;
        sn = getPerturbedNormal(
            mat, geom, onDevice.samplers, hr.prim_id, uv, tng, btng, sn);
      }
      vec4f color = getColor(mat, geom, onDevice.samplers, hr.prim_id, uv);

      float3 xfmDir = (inst.invXfm * float4(ray.dir, 0.f)).xyz();

      // That doesn't work for instances..
      float3 shadedColor{0.f};

      if (rendererState.renderMode == RenderMode::Default) {
        float3 viewDir = -xfmDir;
        float3 lightDir = -xfmDir;
        float3 intensity(1.f);
        shadedColor = evalMaterial(mat,
                                   geom,
                                   onDevice.samplers,
                                   hr.prim_id,
                                   uv, gn, sn,
                                   viewDir,
                                   lightDir,
                                   intensity);
        shadedColor +=
            color.xyz() * rendererState.ambientColor * rendererState.ambientRadiance;
      }
      else if (rendererState.renderMode == RenderMode::Ng)
        shadedColor = (gn + float3(1.f)) * float3(0.5f);
      else if (rendererState.renderMode == RenderMode::Ns)
        shadedColor = (sn + float3(1.f)) * float3(0.5f);
      else if (rendererState.renderMode == RenderMode::Albedo)
        shadedColor = color.xyz();
      else if (rendererState.renderMode == RenderMode::Tangent)
        shadedColor = tng;
      else if (rendererState.renderMode == RenderMode::Bitangent)
        shadedColor = btng;
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

      float a = getOpacity(mat, geom, onDevice.samplers, hr.prim_id, uv);
      surfaceColor += (1.f-surfaceAlpha) * a * shadedColor;
      surfaceAlpha += (1.f-surfaceAlpha) * a;

      if (!hit) {
        result.depth = hr.t;
        result.Ng = gn;
        result.Ns = sn;
        result.albedo = color.xyz();
        result.primId = hr.prim_id;
        result.objId = hr.geom_id;
        result.instId = hr.inst_id;
      }

      hit = true;

      if (surfaceAlpha < 0.999f) {
        ray.tmin = hr.t + 1e-4f;
        hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
      } else {
        ray.tmax = hr.t;
        break;
      }
    }

    // Background
    if (rendererState.envID >= 0 && onDevice.lights[rendererState.envID].visible) {
      auto hdri = onDevice.lights[rendererState.envID].asHDRI;
      float2 uv = toUV(ray.dir);
      result.color = over(float4(surfaceColor, surfaceAlpha), tex2D(hdri.radiance, uv));
    } else {
      result.color = over(float4(surfaceColor, surfaceAlpha), rendererState.bgColor);
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

      hit = true;
    }

    // if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
    //   result.color = float4(1.f) - result.color;
    // }

    return result;
  }

  RendererState rendererState;

  constexpr static bool stochasticRendering{false};
  constexpr static bool supportsTaa{false};
};

} // namespace visionaray
