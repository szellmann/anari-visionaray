// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Pathtrace_impl.h"
#include "for_each.h"

namespace visionaray {

enum RayType { Radiance, Shadow, AO, Miss, None, };

VSNRAY_FUNC
inline float safe_rcp(float f)
{ return f > 0.f ? 1.f/f : 0.f; }

struct ShadeState
{
  float3 baseColor{0.f};
  float3 shadedColor{0.f};
  float3 hitPosOff{0.f};
  float3 sn{0.f};
  int aoSamples{0};
  float aoWeights{0.f};
  float aoCount{0.f};
  BSDFSample bsdfSample{};
  LightSample lightSample{};
  float misWeightNEE{0.f};
  struct {
    RayType rayType{None};
    Ray ray;
  } next;
  struct {
    float light{1.f};
    float ao{1.f};
  } visibility;
};

template<RayType Type>
VSNRAY_FUNC inline void prepareNextRay(ShadeState &shadeState,
                                       ScreenSample &ss,
                                       const Ray &ray,
                                       const RendererState &rendererState,
                                       const dco::World &world)
{
  auto &hitPosOff = shadeState.hitPosOff;
  auto &sn = shadeState.sn;
  auto &bsdfSample = shadeState.bsdfSample;
  auto &lightSample = shadeState.lightSample;
  auto &next = shadeState.next;

  next.rayType = Type;

  if constexpr (Type == Shadow) {
    Ray &shadowRay = next.ray;
    shadowRay.ori = hitPosOff;
    shadowRay.dir = normalize(lightSample.dir);
    shadowRay.tmin = 0.f;
    shadowRay.tmax = lightSample.dist;//-1e-4f; // TODO: bias sample point
    shadowRay.time = ray.time;
    shadowRay.dbg = ray.dbg;
    next.rayType = Shadow;
  }
  else if constexpr (Type == AO) {
    vec3 u, v, w = sn;
    make_orthonormal_basis(u,v,w);
    auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
    vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

    Ray &aoRay = next.ray;
    aoRay.ori = hitPosOff;
    aoRay.dir = dir;
    aoRay.tmin = 0.f;
    aoRay.tmax = rendererState.occlusionDistance;
    aoRay.time = ray.time;
    aoRay.dbg = ray.dbg;
    next.rayType = AO;
  }
  else if constexpr (Type == Radiance) {
    Ray &bsdfRay = next.ray;
    if (dot(bsdfSample.dir,sn) > 0.f)
      bsdfRay.ori = hitPosOff;
    else
      bsdfRay.ori = hitPosOff;
    bsdfRay.dir = normalize(bsdfSample.dir);
    bsdfRay.tmin = 0.f;
    bsdfRay.tmax = 1e31f;
    bsdfRay.time = ray.time;
    bsdfRay.dbg = ray.dbg;
    next.rayType = Radiance;
  }
}

VSNRAY_FUNC
inline void shade(ScreenSample &ss, const Ray &ray, RayType rayType, unsigned worldID,
    const DeviceObjectRegistry &onDevice, const RendererState &rendererState,
    const HitRec &hitRec,
    ShadeState &shadeState,
    PixelSample &result,
    unsigned bounceID)
{
  auto &baseColor = shadeState.baseColor;
  auto &shadedColor = shadeState.shadedColor;
  auto &hitPosOff = shadeState.hitPosOff;
  auto &sn = shadeState.sn;
  auto &aoSamples = shadeState.aoSamples;
  auto &aoWeights = shadeState.aoWeights;
  auto &aoCount = shadeState.aoCount;
  auto &bsdfSample = shadeState.bsdfSample;
  auto &lightSample = shadeState.lightSample;
  auto &misWeightNEE = shadeState.misWeightNEE ;
  auto &next = shadeState.next;
  auto &visibility = shadeState.visibility;

  auto &hr = hitRec.surface;
  auto &hrv = hitRec.volume;
  auto &hrl = hitRec.light;

  next.rayType = Miss;

  dco::World world = onDevice.worlds[worldID];

  if (rayType == Radiance) {

    baseColor = float3{0.f};
    visibility.light = 1.f;
    visibility.ao = 1.f;
    aoSamples = 0;
    aoWeights = 0.f;
    aoCount = 0.f;

    if (!hitRec.hit) {
      shadedColor = float3{0.f};
      next.rayType = Miss;
      return;
    }

    float3 hitPos{0.f};
    float4 color{1.f};

    if (hitRec.lightHit) {
      hitPos = ray.ori + hrl.t * ray.dir;
      const dco::Light &light = getLight(world.allLights, hrl.lightID, onDevice);
      if (light.type == dco::Light::Quad) {
        shadedColor = light.asQuad.intensity(hitPos);
      } else if (light.type == dco::Light::HDRI) {
        shadedColor = light.asHDRI.intensity(ray.dir);
      }
      misWeightNEE = 1.f;

      // Multiply by MIS weight:
      if (bounceID > 0) {
        float lightPDF = 0.f;
        if (light.type == dco::Light::Quad) {
          float A = area(light.asQuad.geometry());
          float ld = length(hitPos-ray.ori);
          float3 L = normalize(hitPos-ray.ori);
          float3 Nl = get_normal(hitRec,light.asQuad.geometry());
          float LdotNl = fabsf(dot(-L,Nl));
          float solidAngle = (LdotNl*A) / (ld*ld);
          lightPDF = 1.f/solidAngle;
        } else if (light.type == dco::Light::HDRI) {
          float3 dir = light.asHDRI.toLocal*ray.dir;
          float2 uv = toUV(dir);
          CDFSample sample = sampleCDF(light.asHDRI.cdf.rows, light.asHDRI.cdf.lastCol,
                                       light.asHDRI.cdf.width, light.asHDRI.cdf.height,
                                       uv.x, uv.y);
          float theta = acosf(clamp(dir.y, -1.0f, 1.0f));
          float sinTheta = sinf(theta);
          if (sinTheta != 0.f) {
            lightPDF = (sample.pdfx * sample.pdfy)
                * (light.asHDRI.cdf.width * light.asHDRI.cdf.height)
                / (2.0f * constants::pi<float>() * constants::pi<float>() * sinTheta);
          }
        }

        float misWeightBSDF = power_heuristic(bsdfSample.pdf,lightPDF/world.numLights);

        shadedColor *= misWeightBSDF;
      }

      next.rayType = Miss;
      return;
    }

    int instID = hitRec.volumeHit ? hrv.instID : hr.inst_id;
    const dco::Instance &inst = onDevice.instances[instID];
    const dco::Group &group = onDevice.groups[inst.groupID];

    dco::AttributeRec attribs = {};

    float3 localHitPos{0.f}, gn{0.f}, tng{0.f}, btng{0.f};
    int primID = -1;
    float eps = 1e-4f;

    float3 viewDir = -normalize(ray.dir);

    if (hitRec.volumeHit) {
      hitPos = ray.ori + hrv.t * ray.dir;
      localHitPos = hrv.isect_pos;
      primID = hrv.primID;
      eps = epsilonFrom(hitPos, ray.dir, hrv.t);

      const dco::Volume &vol = onDevice.volumes[group.volumes[hrv.localID]];

      if (rendererState.gradientShading) {
        float3 P = vol.field.pointToVoxelSpace(localHitPos);
        float3 delta(vol.field.cellSize, vol.field.cellSize, vol.field.cellSize);
        delta *= float3(vol.field.voxelSpaceTransform(0,0),
                        vol.field.voxelSpaceTransform(1,1),
                        vol.field.voxelSpaceTransform(2,2));
        if (sampleGradient(vol.field,P,delta,gn))
          gn = normalize(gn);
      }

      if (rendererState.ambientSamples > 0 && length(gn) < 1e-3f)
        gn = uniform_sample_sphere(ss.random(), ss.random());

      sn = gn;

      mat3 nxfm = getNormalTransform(inst, ray);
      gn = nxfm * gn;
      sn = nxfm * sn;

      sn = faceforward(sn, viewDir, gn);
      gn = faceforward(gn, viewDir, gn);

      color.xyz() = hrv.albedo;

      if (bounceID==0) {
        result.depth = hrv.t;
        result.primId = hrv.primID;
        result.objId = group.objIds[hrv.localID];
        result.instId = inst.userID;
      }
    } else {
      hitPos = ray.ori + hr.t * ray.dir;
      localHitPos = hr.isect_pos;
      primID = hr.prim_id;
      eps = epsilonFrom(hitPos, ray.dir, hr.t);

      const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
      const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

      float2 uv{hr.u,hr.v};
      getNormals(geom, hr.prim_id, localHitPos, uv, gn, sn);

      float3 worldNormal = gn;

      mat3 nxfm = getNormalTransform(inst, ray);
      gn = normalize(nxfm * gn);
      sn = normalize(nxfm * sn);

      sn = faceforward(sn, viewDir, gn);
      gn = faceforward(gn, viewDir, gn);

      attribs = getAttributes(geom,
                              inst,
                              hitPos,
                              worldNormal,
                              localHitPos,
                              gn,
                              hr.prim_id,
                              uv);

      float4 tng4 = getTangent(geom, hr.prim_id, localHitPos, uv);
      if (length(sn) > 0.f && length(tng4.xyz()) > 0.f) {
        tng = tng4.xyz();
        btng = cross(sn, tng) * tng4.w;
        sn = getPerturbedNormal(
            mat, onDevice, attribs, localHitPos, hr.prim_id, tng, btng, sn);
      }
      color = getColor(mat, onDevice, attribs, localHitPos, hr.prim_id);

      if (bounceID==0) {
        result.depth = hr.t;
        result.primId = hr.prim_id;
        result.objId = group.objIds[hr.geom_id];
        result.instId = inst.userID;
      }
    }

    if (bounceID==0) {
      result.Ng = gn;
      result.Ns = sn;
      result.albedo = color.xyz();
    }

    // Compute new origin for future rays spawned from this hit pos,
    // biased by eps:
    hitPosOff = hitPos + sn * eps;

    // Compute motion vector; assume for now the hit was diffuse!
    recti viewport{0,0,(int)ss.frameSize.x,(int)ss.frameSize.y};
    vec3 prevWP, currWP;
    project(prevWP, hitPos, rendererState.prevMV, rendererState.prevPR, viewport);
    project(currWP, hitPos, rendererState.currMV, rendererState.currPR, viewport);

    result.motionVec = float4(prevWP.xy() - currWP.xy(), 0.f, 1.f);

    int lightID = -1;

    if (world.numLights > 0) {
      lightID = uniformSampleOneLight(ss.random, world.numLights);
      const dco::Light &light = getLight(world.allLights, lightID, onDevice);
      lightSample = sampleLight(light, hitPos, ss.random);
    }

    if (rendererState.renderMode == RenderMode::Default) {
      vec3 lightDir = normalize(lightSample.dir);
      vec3 lightIntensity = lightSample.intensity * safe_rcp(lightSample.dist2);
      const float NdotL = fmaxf(0.f,dot(sn,lightDir));

      if (hitRec.volumeHit) {
        if (rendererState.gradientShading && length(gn) > 1e-10f) {
          dco::Material mat = dco::createMaterial();
          mat.type = dco::Material::Matte;
          mat.asMatte.color = dco::createMaterialParamRGB();
          mat.asMatte.color.rgb = hrv.albedo;

          lightSample.f = evalMaterial(mat,
                                       onDevice,
                                       attribs,
                                       localHitPos,
                                       primID,
                                       gn, gn,
                                       tng, btng,
                                       viewDir,
                                       lightDir);
          shadedColor = lightSample.f * lightIntensity * NdotL
            * safe_rcp(lightSample.pdf) * float(world.numLights);
        } else {
          shadedColor = hrv.albedo * lightIntensity
            * safe_rcp(lightSample.pdf) * safe_rcp(lightSample.dist2);
        }

        // isotropic phase function
        bsdfSample.dir = uniform_sample_sphere(ss.random(), ss.random());
        bsdfSample.f = hrv.albedo * float3(1.f);//over 4 PI (cancels)
        bsdfSample.pdf = 1.f;//over 4 PI (cancels)
        bsdfSample.cosT = 1.f;
      } else {
        const auto &mat = onDevice.materials[group.materials[hr.geom_id]];

        lightSample.f = evalMaterial(mat,
                                     onDevice,
                                     attribs,
                                     localHitPos,
                                     primID,
                                     gn, sn,
                                     tng, btng,
                                     viewDir,
                                     lightDir);
        shadedColor = lightSample.f * lightIntensity * NdotL
          * safe_rcp(lightSample.pdf) * float(world.numLights);

        bsdfSample = sampleMaterial(mat,
                                    onDevice,
                                    attribs,
                                    hr.isect_pos,
                                    hr.prim_id,
                                    gn, sn,
                                    tng, btng,
                                    viewDir, ss.random);
      }

      misWeightNEE = 0.f;
      if (world.numLights > 0) {
        const dco::Light &light = getLight(world.allLights, lightID, onDevice);
        if (light.type == dco::Light::Quad || light.type == dco::Light::HDRI) {
          misWeightNEE
            = power_heuristic(lightSample.pdf/world.numLights,bsdfSample.pdf);
        }
        else {
          // sampled a delta light source:
          misWeightNEE = 1.f;
        }
      }
    }
    else if (rendererState.renderMode == RenderMode::PrimitiveId)
      shadedColor = randomColor(result.primId).xyz();
    else if (rendererState.renderMode == RenderMode::WorldPosition)
      shadedColor = attribs.worldPos.xyz();
    else if (rendererState.renderMode == RenderMode::ObjectPosition)
      shadedColor = attribs.objectPos.xyz();
    else if (rendererState.renderMode == RenderMode::Ng)
      shadedColor = (gn + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Ns)
      shadedColor = (sn + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Tangent)
      shadedColor = (tng + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Bitangent)
      shadedColor = (btng + float3(1.f)) * float3(0.5f);
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
      shadedColor = attribs._0.xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute1)
      shadedColor = attribs._1.xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute2)
      shadedColor = attribs._2.xyz();
    else if (rendererState.renderMode == RenderMode::GeometryAttribute3)
      shadedColor = attribs._3.xyz();
    else if (rendererState.renderMode == RenderMode::GeometryColor)
      shadedColor = attribs.color.xyz();

    if (rendererState.renderMode == RenderMode::Default)
      baseColor = color.xyz();
    else
      baseColor = shadedColor;
  }

  // Advance state machine:
  if (rayType == Radiance) {
    // Is there a light? test if we're in shadow:
    if (lightSample.pdf >= 0) {
      prepareNextRay<Shadow>(shadeState,ss,ray,rendererState,world);
      return;
    }

    // No light? test for AO:
    if (aoSamples < rendererState.ambientSamples) {
      prepareNextRay<AO>(shadeState,ss,ray,rendererState,world);
      return;
    }

    // Visibility accounted for? Try doing a bounce:
    if (bsdfSample.pdf >= 0.f) {
      prepareNextRay<Radiance>(shadeState,ss,ray,rendererState,world);
      return;
    }
    return;
  } else if (rayType == Shadow) {
    // Shadow ray? Finalize light visibility term:
    int surfV = hr.hit ? 0 : 1;
    int volV = hitRec.volumeHit ? 0 : 1;

    visibility.light = surfV * volV * hrv.Tr;

    // Test AO after shadow rays:
    if (aoSamples < rendererState.ambientSamples) {
      prepareNextRay<AO>(shadeState,ss,ray,rendererState,world);
      return;
    }

    // Visibility accounted for? Try doing a bounce:
    if (bsdfSample.pdf >= 0.f) {
      prepareNextRay<Radiance>(shadeState,ss,ray,rendererState,world);
      return;
    }
    return;
  } else if (rayType == AO) {
    aoSamples++;

    float weight = fmaxf(0.f, dot(ray.dir,sn));
    aoWeights += weight;
    if (weight > 0.f && hitRec.hit) {
      aoCount += weight;
    }

    // Tested for AO? Check if there are samples left:
    if (aoSamples < rendererState.ambientSamples) {
      prepareNextRay<AO>(shadeState,ss,ray,rendererState,world);
      return;
    }

    // No more samples to compute? Finalize AO visibility term:
    float aoV = 0.f;
    if (aoWeights > 0.f) {
      aoV = 1.f - (aoCount/aoWeights);
    }
    visibility.ao *= aoV;

    // Visibility accounted for? Try doing a bounce:
    if (bsdfSample.pdf >= 0.f) {
      prepareNextRay<Radiance>(shadeState,ss,ray,rendererState,world);
      return;
    }
    return;
  }
}

void VisionarayRendererPathtrace::renderFrame(DevicePointer<DeviceObjectRegistry> onDevicePtr,
                                              DevicePointer<RendererState> rendererStatePtr,
                                              DevicePointer<dco::Frame> framePtr,
                                              DevicePointer<dco::Camera> camPtr,
                                              uint2 size,
                                              SyncContext::SP syncContext,
                                              unsigned worldID, int frameID)
{
#ifdef WITH_CUDA
  cuda::for_each(syncContext->renderingStream, 0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  hip::for_each(syncContext->renderingStream, 0, size.x, 0, size.y,
#else
  parallel::for_each(syncContext->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;
        const auto &cam = *camPtr;

        int pixelID = x + size.x * y;
        Random rng(pixelID, rendererState.accumID);
        ScreenSample ss{x, y, frameID, size, rng};
        Ray ray;
#ifdef _MSC_VER
        uint64_t clock_begin = clock();
#else
        uint64_t clock_begin = clock64();
#endif

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
          if (rendererState.bgImage.width())
            ps.color = tex2D(rendererState.bgImage,float2(xf/size.x,yf/size.y));
          else
            ps.color = rendererState.bgColor;
          ps.depth = 1e31f;
          ps.albedo = float3(0.f);
          ps.motionVec = float4(0,0,0,1);

          if (onDevice.TLSs[worldID].num_primitives() != 0) {

            HitRec firstHit;
            ShadeState shadeState;
            RayType rayType = Radiance;
            float3 throughput{1.f};
            float3 intensity{0.f};
            for (unsigned passID=0, bounceID=0;true;++passID) {
              ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);
              bool shadow = rayType == Shadow || rayType == AO;
              HitRec hitRec = intersectAll(ss, ray, worldID, onDevice, bounceID, shadow);
              // 1. radiance
              // 2. shadow (optional)
              // 3. AO (optional)
              shade(ss, ray, rayType, worldID, onDevice,
                    rendererState,
                    hitRec,
                    shadeState,
                    ps, bounceID);

              if (passID == 0 && bounceID == 0) {
                firstHit = hitRec;
              }

              ray = shadeState.next.ray;
              rayType = shadeState.next.rayType;

              if (rayType == Miss || rayType == Radiance) {
                float3 direct = (shadeState.shadedColor * shadeState.visibility.light);
                float3 ambient= (shadeState.baseColor * rendererState.ambientColor
                        * rendererState.ambientRadiance * shadeState.visibility.ao);
                intensity += throughput * shadeState.misWeightNEE * direct;
                intensity += throughput * ambient;
                throughput *= shadeState.bsdfSample.f
                    * shadeState.bsdfSample.cosT * safe_rcp(shadeState.bsdfSample.pdf);
                bounceID++;
              }

              if (rayType == Miss || bounceID > rendererState.maxBounce) {
                break;
              }

              // Russian roulette
              if (bounceID > 1) {
                float prob = max_element(throughput);
                if (rng.next() > prob)
                  break;
                throughput /= prob;
              }
            }

            if (firstHit.hit) {
              // if we hit an invisible light with a primary ray we render
              // the background color:
              if (!(firstHit.lightHit && !firstHit.light.lightVisible)) {
                ps.color = float4(intensity,1.f);
              }
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
        PixelSample finalSample = closestSample;
        finalSample.color = accumColor*(1.f/spp);
        if (rendererState.taaEnabled)
          frame.fillGBuffer(x, y, finalSample);
        else
          frame.writeSample(x, y, rendererState.accumID, finalSample);
     });
}

} // namespace visionaray
