
#include "frame/for_each.h"
#include "DirectLight_impl.h"

namespace visionaray {

struct HitRec
{
  hit_record<Ray, primitive<unsigned>> surface;
  HitRecordVolume volume;
  bool hit{false};
  bool volumeHit{false};
};

VSNRAY_FUNC
HitRec intersectAll(ScreenSample &ss, const Ray &ray, unsigned worldID,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice)
{
  HitRec hr;
  hr.surface = intersectSurfaces<1>(ss, ray, onDevice, worldID);
  hr.volume  = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice);
  hr.hit = hr.surface.hit || hr.volume.hit;
  hr.volumeHit = hr.volume.hit && (!hr.surface.hit || hr.volume.t < hr.surface.t);
  return hr;
}

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
  bool hdriMiss{false};
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
  auto &hdriMiss = shadeRec.hdriMiss;

  auto &hr = hitRec.surface;
  auto &hrv = hitRec.volume;

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
    float3 xfmDir = ray.dir;
    float2 uv{hr.u,hr.v};

    if (hitRec.volumeHit) {
      hitPos = ray.ori + hrv.t * ray.dir;
      if (rendererState.gradientShading &&
          sampleGradient(onDevice.spatialFields[hrv.fieldID],hitPos,gn)) {
        gn = normalize(gn);
      }

      result.Ng = gn;
      result.Ns = gn;
      result.albedo = hrv.albedo;
    } else {
      result.depth = hr.t;
      result.primId = hr.prim_id;

      const dco::Instance &inst = onDevice.instances[hr.inst_id];
      const dco::Group &group = onDevice.groups[inst.groupID];
      const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

      result.objId = group.objIds[hr.geom_id];
      result.instId = inst.userID;

      hitPos = ray.ori + hr.t * ray.dir;
      gn = getNormal(geom, hr.prim_id, hitPos, uv);
      sn = getShadingNormal(geom, hr.prim_id, hitPos, uv);
      float4 tng4 = getTangent(geom, hr.prim_id, hitPos, uv);
      if (length(sn) > 0.f && length(tng4.xyz()) > 0.f) {
        tng = tng4.xyz();
        btng = cross(sn, tng) * tng4.w;
        sn = getPerturbedNormal(
            mat, geom, onDevice.samplers, hr.prim_id, uv, tng, btng, sn);
      }
      color = getColor(mat, geom, onDevice.samplers, hr.prim_id, uv);

      result.Ng = gn;
      result.Ns = sn;
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

    int instID = hitRec.volumeHit ? hrv.inst_id : hr.inst_id;
    const dco::Instance &inst = onDevice.instances[instID];
    const dco::Group &group = onDevice.groups[inst.groupID];
    light_sample<float> ls;
    vec3f intensity(0.f);
    float dist = 1.f;
    ls.pdf = 0.f;

    if (world.numLights > 0) {
      int lightID = uniformSampleOneLight(ss.random, world.numLights);

      const dco::Light &light = onDevice.lights[world.allLights[lightID]];

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

    if (hitRec.volumeHit) {
      if (rendererState.renderMode == RenderMode::Default) {
        if (rendererState.gradientShading && length(gn) > 1e-10f) {
          dco::Material mat;
          mat.type = dco::Material::Matte;
          mat.asMatte.color.rgb = hrv.albedo;
          dco::Geometry dummyGeom;

          if (ls.pdf > 0.f) {
            shadedColor = evalMaterial(mat,
                                       dummyGeom,
                                       onDevice.samplers, // not used..
                                       UINT_MAX, vec2{}, // primID and uv, not used..
                                       gn, gn,
                                       viewDir, ls.dir,
                                       intensity);
            shadedColor = shadedColor / ls.pdf / (dist*dist);
          }
        }
        else
          shadedColor = hrv.albedo * intensity / ls.pdf / (dist*dist);
      } else if (rendererState.renderMode == RenderMode::Ng) {
        shadedColor = gn;
      } else if (rendererState.renderMode == RenderMode::Ns) {
        shadedColor = sn;
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
      // That doesn't work for instances..
      const auto &inst = onDevice.instances[hr.inst_id];
      const auto &group = onDevice.groups[inst.groupID];
      const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const auto &mat = onDevice.materials[group.materials[hr.geom_id]];
      if (rendererState.renderMode == RenderMode::Default) {
        if (ls.pdf > 0.f) {
          shadedColor = evalMaterial(mat,
                                     geom,
                                     onDevice.samplers,
                                     hr.prim_id,
                                     uv, gn, sn,
                                     viewDir,
                                     ls.dir,
                                     intensity);
          shadedColor = shadedColor / ls.pdf / (dist*dist);
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
    int volV = hitRec.volumeHit ? 0 : 1;

    if (rendererState.ambientSamples > 0 && length(gn) < 1e-3f)
      gn = uniform_sample_sphere(ss.random(), ss.random());

    float aoV = rendererState.ambientSamples == 0 ? 1.f
        : 1.f-computeAO(ss, worldID, onDevice, gn, sn, viewDir, hitPos,
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
        PixelSample firstSample;
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
        if (rendererState.taaEnabled)
          frame.fillGBuffer(x, y, finalSample);
        else
          frame.writeSample(x, y, rendererState.accumID, finalSample);
     });
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(onDevicePtr));
  CUDA_SAFE_CALL(cudaFree(rendererStatePtr));
  CUDA_SAFE_CALL(cudaFree(framePtr));
#endif
}

} // namespace visionaray
