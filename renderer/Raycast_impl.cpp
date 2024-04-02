
#include "frame/for_each.h"
#include "Raycast_impl.h"

namespace visionaray {

VSNRAY_FUNC
inline PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
    const RendererState &rendererState)
{
  PixelSample result;
  result.color = rendererState.bgColor;
  result.depth = 1e31f;

  if (onDevice.TLSs[worldID].num_primitives() == 0)
    return result; // happens eg with TLSs of unsupported objects

  dco::World world = onDevice.worlds[worldID];

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
    vec3f gn = getNormal(geom, hr.prim_id, hitPos, uv);
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
      for (unsigned lightID=0; lightID<world.numLights; ++lightID) {
        const dco::Light &light = onDevice.lights[world.allLights[lightID]];

        light_sample<float> ls;
        vec3f intensity(0.f);
        float dist = 1.f;
        ls.pdf = 0.f;

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

        float3 brdf = evalMaterial(mat,
                                   geom,
                                   onDevice.samplers,
                                   hr.prim_id,
                                   uv, gn, sn,
                                   viewDir,
                                   ls.dir,
                                   intensity);
        shadedColor += brdf / ls.pdf / (dist*dist);
      }

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

void VisionarayRendererRaycast::renderFrame(const dco::Frame &frame,
                                            const dco::Camera &cam,
                                            uint2 size,
                                            VisionarayGlobalState *state,
                                            const VisionarayGlobalState::DeviceObjectRegistry &DD,
                                            const RendererState &rendererState,
                                            unsigned worldID, int frameID)
{
#ifdef WITH_CUDA
  VisionarayGlobalState::DeviceObjectRegistry *onDevicePtr;
  CUDA_SAFE_CALL(cudaMalloc(&onDevicePtr, sizeof(DD)));
  CUDA_SAFE_CALL(cudaMemcpy(onDevicePtr, &DD, sizeof(DD), cudaMemcpyHostToDevice));

  RendererState *rendererStatePtr;
  CUDA_SAFE_CALL(cudaMalloc(&rendererStatePtr, sizeof(rendererState)));
  CUDA_SAFE_CALL(cudaMemcpy(rendererStatePtr,
                            &rendererState,
                            sizeof(rendererState),
                            cudaMemcpyHostToDevice));

  dco::Frame *framePtr;
  CUDA_SAFE_CALL(cudaMalloc(&framePtr, sizeof(frame)));
  CUDA_SAFE_CALL(cudaMemcpy(framePtr, &frame, sizeof(frame), cudaMemcpyHostToDevice));

  cuda::for_each(0, size.x, 0, size.y,
#else
  auto *onDevicePtr = &DD;
  auto *rendererStatePtr = &rendererState;
  auto *framePtr = &frame;
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const VisionarayGlobalState::DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;

        ScreenSample ss{x, y, frameID, size, {/*no RNG*/}};
        Ray ray;

        uint64_t clock_begin = clock64();

        float4 accumColor{0.f};
        PixelSample firstSample;
        int spp = rendererState.pixelSamples;

        for (int sampleID=0; sampleID<spp; ++sampleID) {

          ray = cam.primary_ray(
              ss.random, float(x), float(y), float(size.x), float(size.y));
#if 1
          ray.dbg = ss.debug();
#endif

          PixelSample ps = renderSample(ss,
                  ray,
                  worldID,
                  onDevice,
                  rendererState);
          accumColor += ps.color;
          if (sampleID == 0) {
            firstSample = ps;
          }
        }

        uint64_t clock_end = clock64();
        if (rendererState.heatMapEnabled > 0.f) {
            float t = (clock_end - clock_begin)
                * (rendererState.heatMapScale / spp);
            accumColor = over(vec4f(heatMap(t), .5f), accumColor);
        }

        // Color gets accumulated, depth, IDs, etc. are
        // taken from first sample
        PixelSample finalSample = firstSample;
        finalSample.color = accumColor*(1.f/spp);
        frame.writeSample(x, y, rendererState.accumID, finalSample);
      });
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(onDevicePtr));
#endif
}

} // namespace visionaray
