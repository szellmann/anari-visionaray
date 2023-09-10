#pragma once

#include "renderer/common.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, ScreenSample &ss, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts objCounts) {

    auto debug = [=]() {
      return ss.x == ss.frameSize.x/2 && ss.y == ss.frameSize.y/2;
    };

    // if (debug()) printf("Rendering frame ==== %u\n", rendererState.accumID);

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1.f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    // Need at least one light..
    if (!onDevice.lights || objCounts.lights == 0)
      return result;

    float3 throughput{1.f};

    auto hr = intersect(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {

      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      // TODO: currently, this will arbitrarily pick a volume _or_
      // surface BVH if both are present and do overlap
      if (geom.type != dco::Geometry::Volume) {
        vec3f hitPos = ray.ori + hr.t * ray.dir;
        vec3f gn = getNormal(geom, hr.prim_id, hitPos);

        int lightID = uniformSampleOneLight(ss.random, objCounts.lights);

        light_sample<float> ls;
        vec3f intensity(0.f);
        if (onDevice.lights[lightID].type == dco::Light::Point) {
          ls = onDevice.lights[lightID].asPoint.sample(hitPos+1e-4f, ss.random);
          intensity = onDevice.lights[lightID].asPoint.intensity(hitPos);
        } else if (onDevice.lights[lightID].type == dco::Light::Directional) {
          ls = onDevice.lights[lightID].asDirectional.sample(hitPos+1e-4f, ss.random);
          intensity = onDevice.lights[lightID].asDirectional.intensity(hitPos);
        }

        shade_record<float> sr;
        sr.normal = gn;
        sr.geometric_normal = gn;
        sr.view_dir = -ray.dir;
        sr.tex_color = float3(1.f);
        sr.light_dir = ls.dir;
        sr.light_intensity = intensity;

        // That doesn't work for instances..
        const auto &mat = onDevice.materials[hr.geom_id];
        float3 shadedColor = to_rgb(mat.asMatte.data.shade(sr));

        Ray shadowRay;
        shadowRay.ori = hitPos;
        shadowRay.dir = ls.dir;
        shadowRay.tmin = 1e-4f;
        shadowRay.tmax = ls.dist-1e-4f;
        auto shadowHR = intersect(shadowRay, onDevice.TLSs[worldID]);
        if (shadowHR.hit)
          throughput = float3{0.f};
        else {
          float dist
              = onDevice.lights[lightID].type == dco::Light::Directional ? 1.f : ls.dist;
          throughput *= shadedColor / ls.pdf / (dist*dist);
        }

        result.color = float4(throughput,1.f);
      }
    }

    if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
    }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
