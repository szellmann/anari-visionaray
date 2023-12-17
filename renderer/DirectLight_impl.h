#pragma once

#include "renderer/AO.h"
#include "renderer/common.h"
#include "renderer/VolumeIntegration.h"
#include "sampleCDF.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice) const {

    // if (ss.debug()) printf("Rendering frame ==== %u\n", rendererState.accumID);

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1e31f;
    result.albedo = float3(0.f);
    result.motionVec = float4(0,0,0,1);

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    // Need at least one light..
    //if (!onDevice.lights || objCounts.lights == 0)
    //  return result;

    float3 throughput{1.f};
    float3 baseColor{0.f};
    float3 shadedColor{0.f};
    float3 gn{0.f};
    float3 viewDir{0.f};
    float3 hitPos{0.f};
    bool hit = false;
    bool hdriMiss = false;

    for (unsigned bounceID=0;bounceID<2;++bounceID) {
      auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
      auto hrv = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice);
      const bool volumeHit = hrv.hit && (!hr.hit || hrv.t < hr.t);

      if (bounceID == 0) {

        if (!hr.hit && !hrv.hit) {
          if (rendererState.envID >= 0 && onDevice.lights[rendererState.envID].visible) {
            auto hdri = onDevice.lights[rendererState.envID].asHDRI;
            float2 uv = toUV(ray.dir);
            throughput = tex2D(hdri.radiance, uv);
            hdriMiss = true;
          } else {
            throughput = float3{0.f};
          }
          break;
        }

        hit = true;

        float4 color{1.f};
        float3 xfmDir = ray.dir;
        float2 uv{hr.u,hr.v};

        if (volumeHit) {
          hitPos = ray.ori + hrv.t * ray.dir;
          if (sampleGradient(onDevice.spatialFields[hrv.fieldID],hitPos,gn))
            gn = normalize(gn);

          result.Ng = gn;
          result.Ns = gn;
          result.albedo = hrv.albedo;
        } else {
          result.depth = hr.t;
          result.primId = hr.prim_id;
          result.objId = hr.geom_id;
          result.instId = hr.inst_id;

          const dco::Instance &inst = onDevice.instances[hr.inst_id];
          const dco::Group &group = onDevice.groups[inst.groupID];
          const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
          const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

          hitPos = ray.ori + hr.t * ray.dir;
          gn = getNormal(geom, hr.prim_id, hitPos);
          if (mat.type == dco::Material::Matte && mat.asMatte.samplerID < UINT_MAX) {
            const auto &samp = onDevice.samplers[mat.asMatte.samplerID];
            color = getSample(samp, geom, hr.prim_id, uv);
          } else {
            color = getColor(geom, mat, hr.prim_id, uv);
          }

          result.Ng = gn;
          result.Ns = gn;
          result.albedo = color.xyz();

          xfmDir = (inst.invXfm * float4(ray.dir, 0.f)).xyz();
        }

        viewDir = -xfmDir;

        // Compute motion vector; assume for now the hit was diffuse!
        recti viewport{0,0,(int)ss.frameSize.x,(int)ss.frameSize.y};
        vec3 prevWP, currWP;
        project(prevWP, hitPos, rendererState.prevMV, rendererState.prevPR, viewport);
        project(currWP, hitPos, rendererState.currMV, rendererState.currPR, viewport);

        result.motionVec = float4(prevWP.xy() - currWP.xy(), 0.f, 1.f);

        const dco::Instance &inst = onDevice.instances[hr.inst_id];
        const dco::Group &group = onDevice.groups[inst.groupID];
        light_sample<float> ls;
        vec3f intensity(0.f);
        float dist = 1.f;

        if (group.numLights > 0) {
          int lightID = uniformSampleOneLight(ss.random, group.numLights);

          const dco::Light &light = onDevice.lights[group.lights[lightID]];

          if (light.type == dco::Light::Point) {
            ls = light.asPoint.sample(hitPos+1e-4f, ss.random);
            intensity = light.asPoint.intensity(hitPos);
          } else if (light.type == dco::Light::Directional) {
            ls = light.asDirectional.sample(hitPos+1e-4f, ss.random);
            intensity = light.asDirectional.intensity(hitPos);
          } else if (light.type == dco::Light::HDRI) {
            ls = light.asHDRI.sample(hitPos+1e-4f, ss.random);
            intensity = light.asHDRI.intensity(ls.dir);
          }

          dist = light.type == dco::Light::Directional||dco::Light::HDRI ? 1.f : ls.dist;
        }

        if (volumeHit) {
          shade_record<float> sr;
          sr.normal = gn;
          sr.geometric_normal = gn;
          sr.view_dir = viewDir;
          sr.tex_color = float3(1.f);
          sr.light_dir = normalize(ls.dir);
          sr.light_intensity = intensity;

          dco::Material mat;
          mat.asMatte.data.cd() = from_rgb(hrv.albedo);
          mat.asMatte.data.kd() = 1.f;

          if (rendererState.renderMode == RenderMode::Default) {
            if (rendererState.gradientShading && length(gn) > 1e-10f)
              shadedColor = to_rgb(mat.asMatte.data.shade(sr)) / ls.pdf / (dist*dist);
            else
              shadedColor = hrv.albedo * intensity / ls.pdf / (dist*dist);
          } else if (rendererState.renderMode == RenderMode::Ng) {
            shadedColor = gn;
          } else if (rendererState.renderMode == RenderMode::Albedo) {
            shadedColor = hrv.albedo;
          } else if (rendererState.renderMode == RenderMode::MotionVec) {
            vec2 xy = normalize(result.motionVec.xy());
            float angle = (1.f+ sinf(xy.x)) *.5f;
            float mag = 1.f;//length(result.motionVec.xy());
            vec3 hsv(angle,1.f,mag);
            shadedColor = hsv2rgb(hsv);
          }

          baseColor = hrv.albedo;
        } else {
          shade_record<float> sr;
          sr.normal = gn;
          sr.geometric_normal = gn;
          sr.view_dir = viewDir;
          sr.tex_color = color.xyz();//float3(1.f);
          sr.light_dir = normalize(ls.dir);
          sr.light_intensity = intensity;

          // That doesn't work for instances..
          const auto &inst = onDevice.instances[hr.inst_id];
          const auto &group = onDevice.groups[inst.groupID];
          const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
          const auto &mat = onDevice.materials[group.materials[hr.geom_id]];
          if (rendererState.renderMode == RenderMode::Default)
            shadedColor = to_rgb(mat.asMatte.data.shade(sr)) / ls.pdf / (dist*dist);
          else if (rendererState.renderMode == RenderMode::Ng)
            shadedColor = (gn + float3(1.f)) * float3(0.5f);
          else if (rendererState.renderMode == RenderMode::Albedo)
            shadedColor = color.xyz();
          else if (rendererState.renderMode == RenderMode::MotionVec) {
            vec2 xy = result.motionVec.xy();
            //xy.x /= float(ss.frameSize.x);
            //xy.y /= float(ss.frameSize.y);
            float x = xy.x, y = xy.y;
            vec2 plr = length(xy) < 1e-10f ? vec2(0.f) : vec2(sqrt(x * x + y * y),atan(y / x));
            //float angle = length(xy) < 1e-8f ? 0 : acos(dot(xy, vec2(1,0))/length(xy)) * visionaray::constants::radians_to_degrees<float>();
            //float angle = (plr.y+M_PI*.5f) * visionaray::constants::radians_to_degrees<float>();
            float angle = 180+plr.y * visionaray::constants::radians_to_degrees<float>();
            float mag = plr.x;
            vec3 hsv(angle,1.f,mag);
            shadedColor = hsv2rgb(hsv);
          } else if (rendererState.renderMode == RenderMode::GeometryAttribute0)
            shadedColor = getAttribute(geom, dco::Attribute::_0, hr.prim_id, uv).xyz();
          else if (rendererState.renderMode == RenderMode::GeometryAttribute1)
            shadedColor = getAttribute(geom, dco::Attribute::_1, hr.prim_id, uv).xyz();
          else if (rendererState.renderMode == RenderMode::GeometryAttribute2)
            shadedColor = getAttribute(geom, dco::Attribute::_2, hr.prim_id, uv).xyz();
          else if (rendererState.renderMode == RenderMode::GeometryAttribute3)
            shadedColor = getAttribute(geom, dco::Attribute::_3, hr.prim_id, uv).xyz();
          else if (rendererState.renderMode == RenderMode::GeometryColor)
            shadedColor = getAttribute(geom, dco::Attribute::Color, hr.prim_id, uv).xyz();

          if (rendererState.renderMode == RenderMode::Default)
            baseColor = color.xyz();
          else
            baseColor = shadedColor;
        }

        // Convert primary to shadow ray
        ray.ori = hitPos;
        ray.dir = ls.dir;
        ray.tmin = 1e-4f;
        ray.tmax = ls.dist-1e-4f;
      } else { // bounceID == 1
        int surfV = hr.hit ? 0 : 1;
        int volV = volumeHit ? 0 : 1;
        float aoV = rendererState.ambientSamples == 0 ? 1.f
            : 1.f-computeAO(ss, worldID, onDevice, gn, viewDir, hitPos,
                            rendererState.ambientSamples,
                            rendererState.occlusionDistance);
        // visibility term
        float V = surfV * volV * hrv.Tr;
        throughput *= shadedColor * V
            + (baseColor * rendererState.ambientColor
             * rendererState.ambientRadiance * aoV);
      }
    }

    if (hit || hdriMiss) {
      result.color = float4(throughput,1.f);
    }

    // if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
    //   result.color = float4(1.f) - result.color;
    // }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
