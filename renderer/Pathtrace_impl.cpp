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
  float3 emission{0.f};
  float3 hitPos{0.f};
  float3 gn{0.f}, sn{0.f}, ln{0.f};
  int aoSamples{0};
  float aoWeights{0.f};
  float aoCount{0.f};
  BSDFSample bsdfSample{};
  LightSample lightSample{};
  float misWeightBSDF{1.f};
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
  auto &hitPos = shadeState.hitPos;
  auto &gn = shadeState.gn;
  auto &sn = shadeState.sn;
  auto &ln = shadeState.ln;
  auto &bsdfSample = shadeState.bsdfSample;
  auto &lightSample = shadeState.lightSample;
  auto &next = shadeState.next;

  next.rayType = Type;
  next.ray.ori = offsetRayOrigin(hitPos, gn);

  if constexpr (Type == Shadow) {
    float3 Nl = ln;
    // orient light normal towards shadow ray origin
    if (dot(Nl,-lightSample.dir) < 0.f) Nl = -Nl;

    float3 lightPos = hitPos+lightSample.dir;
    float3 offsetLightPos = offsetRayOrigin(lightPos, Nl);
    float3 lightDir = offsetLightPos-next.ray.ori;
    float d = length(lightDir);

    Ray &shadowRay = next.ray;
    shadowRay.dir = lightDir / d;
    shadowRay.tmin = 0.f;
    shadowRay.tmax = d;
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
    aoRay.dir = dir;
    aoRay.tmin = 0.f;
    aoRay.tmax = rendererState.occlusionDistance;
    aoRay.time = ray.time;
    aoRay.dbg = ray.dbg;
    next.rayType = AO;
  }
  else if constexpr (Type == Radiance) {
    Ray &bsdfRay = next.ray;
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
  auto &emission = shadeState.emission;
  auto &hitPos = shadeState.hitPos;
  auto &gn = shadeState.gn;
  auto &sn = shadeState.sn;
  auto &ln = shadeState.ln;
  auto &aoSamples = shadeState.aoSamples;
  auto &aoWeights = shadeState.aoWeights;
  auto &aoCount = shadeState.aoCount;
  auto &bsdfSample = shadeState.bsdfSample;
  auto &lightSample = shadeState.lightSample;
  auto &misWeightBSDF = shadeState.misWeightBSDF ;
  auto &misWeightNEE = shadeState.misWeightNEE ;
  auto &next = shadeState.next;
  auto &visibility = shadeState.visibility;

  next.rayType = Miss;

  dco::World world = onDevice.worlds[worldID];

  if (rayType == Radiance) {

    baseColor = float3{0.f};
    shadedColor = float3{0.f};
    emission = float3{0.f};
    visibility.light = 1.f;
    visibility.ao = 1.f;
    aoSamples = 0;
    aoWeights = 0.f;
    aoCount = 0.f;

    if (!hitRec.hit) {
      next.rayType = Miss;
      return;
    }

    hitPos = ray.ori + hitRec.t * ray.dir;

    if (hitRec.type == HitRec::Light) {
      const dco::Light &light = getLight(world.allLights, hitRec.objID, onDevice);
      emission = light.radiance(hitPos,ray.dir);
      misWeightNEE = 1.f;

      // Multiply by MIS weight:
      if (bounceID > 0) {
        float lightPDF = 0.f;
        if (light.type == dco::Light::Quad) {
          lightPDF = light.asQuad.pdf(ray,hitPos);
        } else if (light.type == dco::Light::HDRI) {
          lightPDF = light.asHDRI.pdf(ray,hitPos);
        } else if (light.type == dco::Light::Point) {
          lightPDF = light.asPoint.pdf(ray,hitPos);
        }

        misWeightBSDF = power_heuristic(bsdfSample.pdf,lightPDF/world.numLights());
      }

      next.rayType = Miss;
      return;
    }

    const dco::Instance &inst = onDevice.instances[hitRec.instID];
    const dco::Group &group = onDevice.groups[inst.groupID];

    dco::AttributeRec attribs = {};

    float4 color{1.f};
    float3 tng{0.f}, btng{0.f};
    float3 viewDir = -normalize(ray.dir);

    if (hitRec.type == HitRec::Volume) {
      const dco::Volume &vol = onDevice.volumes[group.volumes[hitRec.objID]];

      if (vol.gradientShading) {
        float3 P = vol.field.pointToVoxelSpace(hitRec.localHitPos);
        float3 delta(vol.field.cellSize, vol.field.cellSize, vol.field.cellSize);
        delta *= float3(vol.field.voxelSpaceTransform(0,0),
                        vol.field.voxelSpaceTransform(1,1),
                        vol.field.voxelSpaceTransform(2,2));
        if (sampleGradient(vol.field,P,delta,gn))
          gn = normalize(gn);
      }

      // TODO: this overwrites the (gradient shading) normal?!
      if (rendererState.ambientSamples > 0 && length(gn) < 1e-3f)
        gn = uniform_sample_sphere(ss.random(), ss.random());

      sn = gn;

      mat3 nxfm = getNormalTransform(inst, ray);
      gn = nxfm * gn;
      sn = nxfm * sn;

      if (dot(gn,viewDir) < 0.f) gn = -gn;
      if (dot(sn,viewDir) < 0.f) sn = -sn;

      color.xyz() = hitRec.asVolume.albedo;
    } else {
      const dco::Geometry &geom = onDevice.geometries[group.geoms[hitRec.objID]];
      const dco::Material &mat = onDevice.materials[group.materials[hitRec.objID]];

      float2 uv{hitRec.asSurface.u,hitRec.asSurface.v};
      getNormals(geom, hitRec.primID, hitRec.localHitPos, uv, gn, sn);

      float3 worldNormal = gn;

      mat3 nxfm = getNormalTransform(inst, ray);
      gn = normalize(nxfm * gn);
      sn = normalize(nxfm * sn);

      if (dot(gn,viewDir) < 0.f) gn = -gn;
      if (dot(sn,viewDir) < 0.f) sn = -sn;

      attribs = getAttributes(geom,
                              inst,
                              hitPos,
                              worldNormal,
                              hitRec.localHitPos,
                              gn,
                              hitRec.primID,
                              uv);

      color = getColor(mat, onDevice, attribs, hitRec.localHitPos, hitRec.primID);

      float4 tangent = getTangent(geom, hitRec.primID, hitRec.localHitPos, uv);
      if (length(sn) > 0.f && length(tangent.xyz()) > 0.f) {
        tng = tangent.xyz();
        btng = cross(sn, tng) * tangent.w;
        sn = getPerturbedNormal(
            mat, onDevice, attribs, hitRec.localHitPos, hitRec.primID, tng, btng, sn);
      }

      emission = getEmission(mat, onDevice, attribs, hitRec.localHitPos, hitRec.primID);
      if (bounceID > 0 && rgb_to_luminance(emission) > FLT_MIN) {
        float A_prim = 0.f;
        if (geom.type == dco::Geometry::Triangle) {
          auto triangle = geom.as<dco::Triangle>(hitRec.primID);
          const mat4 &xfm = inst.xfms[0];
          float3 v1 = (xfm * float4(triangle.v1, 1.f)).xyz();
          float3 v2 = (xfm * float4(triangle.v1 + triangle.e1, 1.f)).xyz();
          float3 v3 = (xfm * float4(triangle.v1 + triangle.e2, 1.f)).xyz();
          A_prim = area(dco::Triangle(v1, v2 - v1, v3 - v1));
        } else if (geom.type == dco::Geometry::Sphere) {
          auto sphere = geom.as<dco::Sphere>(hitRec.primID);
          // TODO: transform? And if so, how do we handle non-uniform scale?
          A_prim = area(sphere);
        }

        float ld = length(hitPos-ray.ori);
        float3 L = normalize(hitPos-ray.ori);
        float LdotNl = dot(-L,gn);
        if (LdotNl < 0.f) {
          L = -L;
          LdotNl = -LdotNl;
        }

        float areaPDF = A_prim > 0.f ? 1.f / (geom.primitives.len * A_prim) : 0.f;
        //float lightPDF = LdotNl > 1e-12f ? areaPDF * (ld * ld) / LdotNl : 0.f;
        float lightPDF = areaPDF * (ld * ld) / LdotNl;
        misWeightBSDF = power_heuristic(bsdfSample.pdf,lightPDF/world.numLights());
      }
    }

    if (bounceID==0) {
      result.depth = hitRec.t;
      result.primId = hitRec.primID;
      result.objId = group.objIds[hitRec.objID];
      result.instId = inst.userID;
      result.Ng = gn;
      result.Ns = sn;
      result.albedo = color.xyz();
    }

    // Compute motion vector; assume for now the hit was diffuse!
    recti viewport{0,0,(int)ss.frameSize.x,(int)ss.frameSize.y};
    vec3 prevWP, currWP;
    project(prevWP, hitPos, rendererState.prevMV, rendererState.prevPR, viewport);
    project(currWP, hitPos, rendererState.currMV, rendererState.currPR, viewport);

    result.motionVec = float4(prevWP.xy() - currWP.xy(), 0.f, 1.f);

    auto pickedLight = world.lightSampler.sample(ss.random);
    unsigned lightID = pickedLight.lightID;
    float lWeight = pickedLight.pdf;

    if (dco::validHandle(lightID)) {
      const dco::LightRef &lightRef = world.allLights[lightID];
      lightSample = sampleLight(onDevice, lightRef, hitPos, ss.random, ray.debug());
      ln = lightSample.Nl;
    }

    if (rendererState.renderMode == RenderMode::Default) {
      vec3 lightDir = normalize(lightSample.dir);
      const float NdotL = fmaxf(0.f,dot(sn,lightDir));

      bool prevBSDFSAmpleWasSpecular = bsdfSample.isSpecular;
      float bsdfPDF = 0.f;

      const float lightPDF = lightSample.pdf * lWeight;

      if (hitRec.type == HitRec::Volume)
        shadedColor = hitRec.asVolume.albedo;
      else
        shadedColor = float3(1,1,1);

      const auto &mat = onDevice.materials[group.materials[hitRec.objID]];
      if (mat.type != dco::Material::Unknown) {
        lightSample.f = evalMaterial(mat,
                                     onDevice,
                                     attribs,
                                     hitRec.localHitPos,
                                     hitRec.primID,
                                     gn, sn,
                                     tng, btng,
                                     viewDir,
                                     lightDir,
                                     &bsdfPDF);
        shadedColor *= lightSample.f;
      }
      shadedColor *= lightSample.Le * NdotL * safe_rcp(lightPDF);

      if (hitRec.type == HitRec::Volume) {
        // isotropic phase function
        bsdfSample.dir = uniform_sample_sphere(ss.random(), ss.random());
        bsdfSample.f = hitRec.asVolume.albedo * float3(1.f);//over 4 PI (cancels)
        bsdfSample.pdf = 1.f;//over 4 PI (cancels)
        bsdfSample.cosT = 1.f;
      } else {
        bsdfSample = sampleMaterial(mat,
                                    onDevice,
                                    attribs,
                                    hitRec.localHitPos,
                                    hitRec.primID,
                                    gn, sn,
                                    tng, btng,
                                    viewDir, ss.random);
      }

      misWeightNEE = 0.f;
      if (dco::validHandle(lightID) && !prevBSDFSAmpleWasSpecular && lightPDF > 0.f) {
        const dco::Light &light = getLight(world.allLights, lightID, onDevice);
        if (light.isAreaLight()) {
          misWeightNEE = power_heuristic(lightPDF,bsdfPDF/world.numLights());
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
    int surfV = (hitRec.hit && hitRec.type==HitRec::Surface) ? 0 : 1;
    int volV = (hitRec.hit && hitRec.type==HitRec::Volume) ? 0 : 1;
    float volTr = (hitRec.hit && hitRec.type==HitRec::Volume) ?
        volV*hitRec.asVolume.Tr : 1.f;

    visibility.light = surfV * volTr;

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
                intensity += throughput * shadeState.misWeightBSDF * shadeState.emission;
                intensity += throughput * ambient;
                throughput *= shadeState.bsdfSample.f
                    * shadeState.bsdfSample.cosT * safe_rcp(shadeState.bsdfSample.pdf);
                bounceID++;
              }

              float tpmax = max_element(throughput);

              if (rayType == Miss || bounceID > rendererState.maxBounce ||
                          tpmax < FLT_MIN) {
                break;
              }

              // Russian roulette
              if (bounceID > 1) {
                float prob = tpmax;
                if (rng.next() > prob)
                  break;
                throughput /= prob;
              }
            }

            if (firstHit.hit) {
              // if we hit an invisible light with a primary ray we render
              // the background color:
              if (!(firstHit.type==HitRec::Light && !firstHit.asLight.visible)) {
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
