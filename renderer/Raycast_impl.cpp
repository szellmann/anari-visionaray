
#include "for_each.h"
#include "Raycast_impl.h"

namespace visionaray {

VSNRAY_FUNC
inline PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
    const DeviceObjectRegistry &onDevice, const RendererState &rendererState)
{
  float4 bgColor = rendererState.bgColor;
  if (rendererState.bgImage.width())
    bgColor = tex2D(rendererState.bgImage,
                    float2(ss.x/float(ss.frameSize.x),ss.y/float(ss.frameSize.y)));

  PixelSample result;
  result.color = bgColor;
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
    vec3f localHitPos = hr.isect_pos;
    vec2f uv{hr.u,hr.v};
    vec3f gn, sn;

    float4 attribs[5];
    for (int i=0; i<5; ++i) {
      attribs[i] = getAttribute(geom, inst, (dco::Attribute)i, hr.prim_id, uv);
    }

    getNormals(geom, hr.prim_id, localHitPos, uv, gn, sn);

    mat3 nxfm = getNormalTransform(inst, ray);
    gn = nxfm * gn;
    sn = nxfm * sn;

    vec3f tng{0.f};
    vec3f btng{0.f};
    float4 tng4 = getTangent(geom, hr.prim_id, localHitPos, uv);
    if (length(sn) > 0.f && length(tng4.xyz()) > 0.f) {
      tng = tng4.xyz();
      btng = cross(sn, tng) * tng4.w;
      sn = getPerturbedNormal(
          mat, onDevice.samplers, attribs, hr.prim_id, tng, btng, sn);
    }
    vec4f color = getColor(mat, onDevice.samplers, attribs, hr.prim_id);

    // That doesn't work for instances..
    float3 shadedColor{0.f};

    if (rendererState.renderMode == RenderMode::Default) {
      float3 viewDir = -ray.dir;
      for (unsigned lightID=0; lightID<world.numLights; ++lightID) {
        const dco::Light &light = onDevice.lights[world.allLights[lightID]];

        LightSample ls = sampleLight(light, hitPos, ss.random);

        float3 brdf = evalMaterial(mat,
                                   onDevice.samplers,
                                   attribs,
                                   hr.prim_id,
                                   gn, sn,
                                   normalize(viewDir),
                                   normalize(ls.dir),
                                   ls.intensity);
        shadedColor += brdf / ls.pdf / ls.dist2;
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
      shadedColor = attribs[(int)dco::Attribute::_0].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute1)
      shadedColor = attribs[(int)dco::Attribute::_1].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute2)
      shadedColor = attribs[(int)dco::Attribute::_2].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute3)
      shadedColor = attribs[(int)dco::Attribute::_3].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryColor)
      shadedColor = attribs[(int)dco::Attribute::Color].xyz();


    float a = getOpacity(mat, onDevice.samplers, attribs, hr.prim_id);
    surfaceColor += (1.f-surfaceAlpha) * a * shadedColor;
    surfaceAlpha += (1.f-surfaceAlpha) * a;

    if (!hit) {
      result.depth = hr.t;
      result.Ng = gn;
      result.Ns = sn;
      result.albedo = color.xyz();
      result.primId = hr.prim_id;
      result.objId = group.objIds[hr.geom_id];
      result.instId = inst.userID;
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
    result.color = over(float4(surfaceColor, surfaceAlpha),
                        float4(hdri.intensity(ray.dir), 1.0f));
  } else {
    result.color = over(float4(surfaceColor, surfaceAlpha), bgColor);
  }

  auto hrv = intersectVolumeBounds(ray, onDevice.TLSs[worldID]);

  if (hrv.hit) {
    const auto &inst = onDevice.instances[hrv.instID];
    const auto &group = onDevice.groups[inst.groupID];
    const dco::Volume &vol = onDevice.volumes[group.volumes[hrv.volID]];

    float3 color(0.f);
    float alpha = 0.f;

    rayMarchVolume(ss, ray, vol, color, alpha);
    result.color = over(float4(color,alpha), result.color);
    result.Ng = float3{}; // TODO: gradient
    result.Ns = float3{}; // TODO..
    result.albedo = float3{}; // TODO..
    result.objId = group.objIds[hrv.volID];
    result.instId = inst.userID;

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
                                            const DeviceObjectRegistry &DD,
                                            const RendererState &rendererState,
                                            unsigned worldID, int frameID)
{
#ifdef WITH_CUDA
  DeviceObjectRegistry *onDevicePtr;
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
#elif defined(WITH_HIP)
  DeviceObjectRegistry *onDevicePtr;
  HIP_SAFE_CALL(hipMalloc(&onDevicePtr, sizeof(DD)));
  HIP_SAFE_CALL(hipMemcpy(onDevicePtr, &DD, sizeof(DD), hipMemcpyHostToDevice));

  RendererState *rendererStatePtr;
  HIP_SAFE_CALL(hipMalloc(&rendererStatePtr, sizeof(rendererState)));
  HIP_SAFE_CALL(hipMemcpy(rendererStatePtr,
                          &rendererState,
                          sizeof(rendererState),
                          hipMemcpyHostToDevice));

  dco::Frame *framePtr;
  HIP_SAFE_CALL(hipMalloc(&framePtr, sizeof(frame)));
  HIP_SAFE_CALL(hipMemcpy(framePtr, &frame, sizeof(frame), hipMemcpyHostToDevice));

  hip::for_each(0, size.x, 0, size.y,
#else
  auto *onDevicePtr = &DD;
  auto *rendererStatePtr = &rendererState;
  auto *framePtr = &frame;
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;

        ScreenSample ss{x, y, frameID, size, {/*no RNG*/}};
        Ray ray;

#ifdef _MSC_VER
        uint64_t clock_begin = clock();
#else
        uint64_t clock_begin = clock64();
#endif

        float4 accumColor{0.f};
        PixelSample firstSample;
        int spp = rendererState.pixelSamples;

        for (int sampleID=0; sampleID<spp; ++sampleID) {

          ray = cam.primary_ray(
              ss.random, float(x), float(y), float(size.x), float(size.y));
#if 1
          ray.dbg = ss.debug();
#endif

          ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);

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

#ifdef _MSC_VER
        uint64_t clock_end = clock();
#else
        uint64_t clock_end = clock64();
#endif
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
  CUDA_SAFE_CALL(cudaFree(rendererStatePtr));
  CUDA_SAFE_CALL(cudaFree(framePtr));
#elif defined(WITH_HIP)
  HIP_SAFE_CALL(hipFree(onDevicePtr));
  HIP_SAFE_CALL(hipFree(rendererStatePtr));
  HIP_SAFE_CALL(hipFree(framePtr));
#endif
}

} // namespace visionaray
