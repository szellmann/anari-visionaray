#pragma once

#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts objCounts) {

    // if (ss.debug()) printf("Rendering frame ==== %u\n", rendererState.accumID);

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1.f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    // Need at least one light..
    if (!onDevice.lights || objCounts.lights == 0)
      return result;

    float3 throughput{1.f};
    float3 shadedColor{0.f};

    for (unsigned bounceID=0;bounceID<2;++bounceID) {
      auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
      auto hrv = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice);
      const bool volumeHit = hrv.hit && (!hr.hit || hrv.t < hr.t);

      if (bounceID == 0) {

        if (!hr.hit && !hrv.hit) {
          throughput = float3{0.f};
          break;
        }

        float3 gn{0.f};
        float3 hitPos{0.f};

        if (volumeHit) {
          hitPos = ray.ori + hrv.t * ray.dir;
        } else {
          auto inst = onDevice.instances[hr.inst_id];
          const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

          hitPos = ray.ori + hr.t * ray.dir;
          gn = getNormal(geom, hr.prim_id, hitPos);
        }

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

        if (volumeHit) {
          shadedColor = hrv.albedo * intensity;
        } else {
          shade_record<float> sr;
          sr.normal = gn;
          sr.geometric_normal = gn;
          sr.view_dir = -ray.dir;
          sr.tex_color = float3(1.f);
          sr.light_dir = ls.dir;
          sr.light_intensity = intensity;

          float dist
              = onDevice.lights[lightID].type == dco::Light::Directional ? 1.f : ls.dist;

          // That doesn't work for instances..
          auto inst = onDevice.instances[hr.inst_id];
          const auto &mat = onDevice.groups[inst.groupID].materials[hr.geom_id];
          shadedColor = to_rgb(mat.asMatte.data.shade(sr)) / ls.pdf / (dist*dist);
        }

        // Convert primary to shadow ray
        ray.ori = hitPos;
        ray.dir = ls.dir;
        ray.tmin = 1e-4f;
        ray.tmax = ls.dist-1e-4f;
      } else { // bounceID == 1
        if (!hr.hit && !volumeHit) {
          throughput *= shadedColor;
        } else if (volumeHit) {
          throughput *= hrv.Tr;
        } else {
          throughput = float3{0.f};
        }
      }
    }

    result.color = float4(throughput,1.f);

    if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
    }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
