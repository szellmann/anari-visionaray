
#include "DirectLight_impl.h"
#include "for_each.h"

namespace visionaray {

struct ShadeRec
{
  float3 throughput{1.f};
  float3 baseColor{0.f};
  float3 shadedColor{0.f};
  float3 gn{0.f};
  float3 sn{0.f};
  float3 tng{0.f};
  float3 btng{0.f};
  float3 viewDir{0.f};
  float3 hitPos{0.f};
  float4 attribs[5];
  bool hdriMiss{false};
  float eps{1e-4f};
};

VSNRAY_FUNC
bool shade(ScreenSample &ss, Ray &ray, unsigned worldID,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
    const RendererState &rendererState,
    const HitRec &hitRec,
    ShadeRec &shadeRec,
    PixelSample &result,
    unsigned bounceID)
{
  auto &throughput = shadeRec.throughput;
  auto &baseColor = shadeRec.baseColor;
  auto &shadedColor = shadeRec.shadedColor;
  auto &gn = shadeRec.gn;
  auto &sn = shadeRec.sn;
  auto &tng = shadeRec.tng;
  auto &btng = shadeRec.btng;
  auto &viewDir = shadeRec.viewDir;
  auto &hitPos = shadeRec.hitPos;
  auto &attribs = shadeRec.attribs;
  auto &hdriMiss = shadeRec.hdriMiss;
  auto &eps = shadeRec.eps;

  auto &hr = hitRec.surface;
  auto &hrv = hitRec.volume;
  auto &hrl = hitRec.light;

  dco::World world = onDevice.worlds[worldID];

  if (bounceID == 0) {

    if (!hitRec.hit) {
      if (rendererState.envID >= 0 && onDevice.lights[rendererState.envID].visible) {
        auto hdri = onDevice.lights[rendererState.envID].asHDRI;
        float2 uv = toUV(ray.dir);
        // TODO: type not supported with cuda?!
        throughput = tex2D(hdri.radiance, uv).xyz();
        hdriMiss = true;
      } else {
        throughput = float3{0.f};
      }
      return false;
    }

    float4 color{1.f};
    float2 uv{hr.u,hr.v};

    if (hitRec.lightHit) {
      hitPos = ray.ori + hrl.t * ray.dir;
      const dco::Light &light = onDevice.lights[world.allLights[hrl.lightID]];
      if (light.type == dco::Light::Quad)
        throughput = light.asQuad.intensity(hitPos);
      hdriMiss = true; // TODO?!
      return false;
    }

    int instID = hitRec.volumeHit ? hrv.instID : hr.inst_id;
    const dco::Instance &inst = onDevice.instances[instID];
    const dco::Group &group = onDevice.groups[inst.groupID];

    if (hitRec.volumeHit) {
      hitPos = ray.ori + hrv.t * ray.dir;
      eps = epsilonFrom(hitPos, ray.dir, hrv.t);
      viewDir = -ray.dir;

      const dco::Volume &vol = onDevice.volumes[group.volumes[hrv.volID]];

      if (rendererState.gradientShading) {
        if (sampleGradient(vol.field,hitPos,gn))
          gn = normalize(gn);
      }

      if (rendererState.ambientSamples > 0 && length(gn) < 1e-3f)
        gn = uniform_sample_sphere(ss.random(), ss.random());

      color.xyz() = hrv.albedo;

      result.depth = hrv.t;
      result.objId = group.objIds[hrv.volID];
      result.instId = inst.userID;
    } else {
      result.depth = hr.t;
      result.primId = hr.prim_id;

      const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

      result.objId = group.objIds[hr.geom_id];
      result.instId = inst.userID;

      hitPos = ray.ori + hr.t * ray.dir;
      eps = epsilonFrom(hitPos, ray.dir, hr.t);

      for (int i=0; i<5; ++i) {
        attribs[i] = getAttribute(geom, (dco::Attribute)i, hr.prim_id, uv);
      }

      viewDir = -ray.dir;

      float3 localHitPos = hr.isect_pos;
      getNormals(geom, hr.prim_id, localHitPos, uv, gn, sn);

      mat3 nxfm = getNormalTransform(inst, ray);
      gn = nxfm * gn;
      sn = nxfm * sn;

      sn = faceforward(sn, viewDir, gn);

      float4 tng4 = getTangent(geom, hr.prim_id, localHitPos, uv);
      if (length(sn) > 0.f && length(tng4.xyz()) > 0.f) {
        tng = tng4.xyz();
        btng = cross(sn, tng) * tng4.w;
        sn = getPerturbedNormal(
            mat, onDevice.samplers, attribs, hr.prim_id, tng, btng, sn);
      }
      color = getColor(mat, onDevice.samplers, attribs, hr.prim_id);
    }

    result.Ng = gn;
    result.Ns = sn;
    result.albedo = color.xyz();

    // Compute motion vector; assume for now the hit was diffuse!
    recti viewport{0,0,(int)ss.frameSize.x,(int)ss.frameSize.y};
    vec3 prevWP, currWP;
    project(prevWP, hitPos, rendererState.prevMV, rendererState.prevPR, viewport);
    project(currWP, hitPos, rendererState.currMV, rendererState.currPR, viewport);

    result.motionVec = float4(prevWP.xy() - currWP.xy(), 0.f, 1.f);

    LightSample ls;
    memset(&ls, 0, sizeof(ls));

    if (world.numLights > 0) {
      int lightID = uniformSampleOneLight(ss.random, world.numLights);
      const dco::Light &light = onDevice.lights[world.allLights[lightID]];
      ls = sampleLight(light, hitPos, ss.random);
    }

    if (rendererState.renderMode == RenderMode::Default) {
      auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
      if (hitRec.volumeHit) {
        if (rendererState.gradientShading && length(gn) > 1e-10f) {
          dco::Material mat;
          mat.type = dco::Material::Matte;
          mat.asMatte.color.rgb = hrv.albedo;

          shadedColor = evalMaterial(mat,
                                     onDevice.samplers, // not used..
                                     nullptr, // attribs, not used..
                                     UINT_MAX, // primID, not used..
                                     gn, gn,
                                     viewDir,
                                     ls.dir,
                                     ls.intensity);
          shadedColor = shadedColor * safe_rcp(ls.pdf) / (ls.dist*ls.dist);
        }
        else
          shadedColor = hrv.albedo * ls.intensity * safe_rcp(ls.pdf) / (ls.dist*ls.dist);
      } else {
        const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
        const auto &mat = onDevice.materials[group.materials[hr.geom_id]];

        shadedColor = evalMaterial(mat,
                                   onDevice.samplers,
                                   attribs,
                                   hr.prim_id,
                                   gn, sn,
                                   viewDir,
                                   ls.dir,
                                   ls.intensity);
        shadedColor = shadedColor * safe_rcp(ls.pdf) / (ls.dist*ls.dist);
      }
    }
    else if (rendererState.renderMode == RenderMode::Ng)
      shadedColor = (gn + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Ns)
      shadedColor = (sn + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Tangent)
      shadedColor = tng;
    else if (rendererState.renderMode == RenderMode::Bitangent)
      shadedColor = btng;
    else if (rendererState.renderMode == RenderMode::Albedo)
      shadedColor = color.xyz();
    else if (rendererState.renderMode == RenderMode::MotionVec) {
      vec2 xy = result.motionVec.xy();
      float x = xy.x, y = xy.y;
      vec2 plr = length(xy) < 1e-10f ? vec2(0.f) : vec2(sqrt(x * x + y * y),atan(y / x));
      float angle = 180+plr.y * visionaray::constants::radians_to_degrees<float>();
      float mag = plr.x;
      vec3 hsv(angle,1.f,mag);
      shadedColor = hsv2rgb(hsv);
    } else if (rendererState.renderMode == RenderMode::GeometryAttribute0)
      shadedColor = attribs[(int)dco::Attribute::_0].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute1)
      shadedColor = attribs[(int)dco::Attribute::_1].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute2)
      shadedColor = attribs[(int)dco::Attribute::_2].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute3)
      shadedColor = attribs[(int)dco::Attribute::_3].xyz();
    else if (rendererState.renderMode == RenderMode::GeometryColor)
      shadedColor = attribs[(int)dco::Attribute::Color].xyz();

    if (rendererState.renderMode == RenderMode::Default)
      baseColor = color.xyz();
    else
      baseColor = shadedColor;

    // Convert primary to shadow ray
    ray.ori = hitPos + sn * eps;
    ray.dir = normalize(ls.dir);
    ray.tmin = 0.f;
    ray.tmax = ls.dist;//-1e-4f; // TODO: bias sample point
  } else { // bounceID == 1
    int surfV = hr.hit ? 0 : 1;
    int volV = hitRec.volumeHit ? 0 : 1;

    float aoV = rendererState.ambientSamples == 0 ? 1.f
        : 1.f-computeAO(ss, worldID, onDevice, gn, sn, viewDir, hitPos, ray.time, eps,
                        rendererState.ambientSamples,
                        rendererState.occlusionDistance);
    // visibility term
    float V = surfV * volV * hrv.Tr;
    throughput *= shadedColor * V
        + (baseColor * rendererState.ambientColor
         * rendererState.ambientRadiance * aoV);
  }
  return true;
}

void VisionarayRendererDirectLight::renderFrame(const dco::Frame &frame,
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
#elif defined(WITH_HIP)
  VisionarayGlobalState::DeviceObjectRegistry *onDevicePtr;
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

        const VisionarayGlobalState::DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;

        int pixelID = x + size.x * y;
        Random rng(pixelID, rendererState.accumID);
        ScreenSample ss{x, y, frameID, size, rng};
        Ray ray;

        uint64_t clock_begin = clock64();

        float4 accumColor{0.f};
        PixelSample closestSample;
        closestSample.depth = 1e31f;
        int spp = rendererState.pixelSamples;

        for (int sampleID=0; sampleID<spp; ++sampleID) {

          // jitter pixel sample
          float xf(x), yf(y);
          vec2f jitter(ss.random() - .5f, ss.random() - .5f);
          xf += jitter.x;
          yf += jitter.y;

          ray = cam.primary_ray(ss.random, xf, yf, float(size.x), float(size.y));

#if 1
          ray.dbg = ss.debug();
#endif

          // if (ss.debug()) printf("Rendering frame ==== %u\n", rendererState.accumID);

          PixelSample ps;
          ps.color = rendererState.bgColor;
          ps.depth = 1e31f;
          ps.albedo = float3(0.f);
          ps.motionVec = float4(0,0,0,1);

          if (onDevice.TLSs[worldID].num_primitives() != 0) {

            HitRec firstHit;
            ShadeRec shadeRec;
            for (unsigned bounceID=0;bounceID<2;++bounceID) {
              ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);
              HitRec hitRec = intersectAll(ss, ray, worldID, onDevice);
              if (!shade(ss, ray, worldID, onDevice,
                    rendererState,
                    hitRec,
                    shadeRec,
                    ps,
                    bounceID)) {
                break;
              }

              if (bounceID == 0) {
                firstHit = hitRec;
              }
            }

            if (firstHit.hit || shadeRec.hdriMiss) {
              ps.color = float4(shadeRec.throughput,1.f);
            }

            // if (ss.x == ss.frameSize.x/2 || ss.y == ss.frameSize.y/2) {
            //   ps.color = float4(1.f) - ps.color;
            // }
          }

          accumColor += ps.color;
          if (ps.depth < closestSample.depth) {
            closestSample = ps;
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
        PixelSample finalSample = closestSample;
        finalSample.color = accumColor*(1.f/spp);
        if (rendererState.taaEnabled)
          frame.fillGBuffer(x, y, finalSample);
        else
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
