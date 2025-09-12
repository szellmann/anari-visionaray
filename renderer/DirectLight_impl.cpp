// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "DirectLight_impl.h"
#include "for_each.h"

namespace visionaray {

enum RayType { Radiance, Shadow, AO, None };

struct ShadeState
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
  int aoSamples{0};
  float aoWeights{0.f};
  float aoCount{0.f};
  struct {
    RayType rayType;
    Ray ray;
  } next;
};

VSNRAY_FUNC
inline void shade(ScreenSample &ss, const Ray &ray, RayType rayType, unsigned worldID,
    const DeviceObjectRegistry &onDevice, const RendererState &rendererState,
    const HitRec &hitRec,
    ShadeState &shadeState,
    PixelSample &result)
{
  auto &throughput = shadeState.throughput;
  auto &baseColor = shadeState.baseColor;
  auto &shadedColor = shadeState.shadedColor;
  auto &gn = shadeState.gn;
  auto &sn = shadeState.sn;
  auto &tng = shadeState.tng;
  auto &btng = shadeState.btng;
  auto &viewDir = shadeState.viewDir;
  auto &hitPos = shadeState.hitPos;
  auto &attribs = shadeState.attribs;
  auto &hdriMiss = shadeState.hdriMiss;
  auto &eps = shadeState.eps;
  auto &aoSamples = shadeState.aoSamples;
  auto &aoWeights = shadeState.aoWeights;
  auto &aoCount = shadeState.aoCount;
  auto &next = shadeState.next;

  auto &hr = hitRec.surface;
  auto &hrv = hitRec.volume;
  auto &hrl = hitRec.light;

  next.rayType = None;

  dco::World world = onDevice.worlds[worldID];

  if (rayType == Radiance) {

    if (!hitRec.hit) {
      if (rendererState.envID >= 0 && onDevice.lights[rendererState.envID].visible) {
        auto hdri = onDevice.lights[rendererState.envID].asHDRI;
        throughput = hdri.intensity(ray.dir);
        hdriMiss = true;
      } else {
        throughput = float3{0.f};
      }
      return;
    }

    float4 color{1.f};
    float2 uv{hr.u,hr.v};

    if (hitRec.lightHit) {
      hitPos = ray.ori + hrl.t * ray.dir;
      const dco::Light &light = getLight(world.allLights, hrl.lightID, onDevice);
      if (light.type == dco::Light::Quad)
        throughput = light.asQuad.intensity(hitPos);
      hdriMiss = true; // TODO?!
      return;
    }

    int instID = hitRec.volumeHit ? hrv.instID : hr.inst_id;
    const dco::Instance &inst = onDevice.instances[instID];
    const dco::Group &group = onDevice.groups[inst.groupID];

    viewDir = -ray.dir;

    if (hitRec.volumeHit) {
      hitPos = ray.ori + hrv.t * ray.dir;
      eps = epsilonFrom(hitPos, ray.dir, hrv.t);

      float3 localHitPos = hrv.isect_pos;

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

      color.xyz() = hrv.albedo;

      result.depth = hrv.t;
      result.primId = hrv.primID;
      result.objId = group.objIds[hrv.localID];
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
        attribs[i] = getAttribute(geom, inst, (dco::Attribute)i, hr.prim_id, uv);
      }

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
            mat, onDevice, attribs, localHitPos, hr.prim_id, tng, btng, sn);
      }
      color = getColor(mat, onDevice, attribs, localHitPos, hr.prim_id);
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
      const dco::Light &light = getLight(world.allLights, lightID, onDevice);
      ls = sampleLight(light, hitPos, ss.random);
    }

    if (rendererState.renderMode == RenderMode::Default) {
      auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
      if (hitRec.volumeHit) {
        if (rendererState.gradientShading && length(gn) > 1e-10f) {
          dco::Material mat = dco::createMaterial();
          mat.type = dco::Material::Matte;
          mat.asMatte.color = dco::createMaterialParamRGB();
          mat.asMatte.color.rgb = hrv.albedo;

          shadedColor = evalMaterial(mat,
                                     onDevice,
                                     nullptr, // attribs, not used..
                                     float3(0.f), // objPos, not used..
                                     UINT_MAX, // primID, not used..
                                     gn, gn,
                                     tng, btng,
                                     normalize(viewDir),
                                     normalize(ls.dir),
                                     ls.intensity * safe_rcp(ls.dist2));
          shadedColor = shadedColor * safe_rcp(ls.pdf) * float(world.numLights);
        }
        else
          shadedColor = hrv.albedo * ls.intensity * safe_rcp(ls.pdf) * safe_rcp(ls.dist2);
      } else {
        const auto &geom = onDevice.geometries[group.geoms[hr.geom_id]];
        const auto &mat = onDevice.materials[group.materials[hr.geom_id]];

        shadedColor = evalMaterial(mat,
                                   onDevice,
                                   attribs,
                                   hr.isect_pos,
                                   hr.prim_id,
                                   gn, sn,
                                   tng, btng,
                                   normalize(viewDir),
                                   normalize(ls.dir),
                                   ls.intensity * safe_rcp(ls.dist2));
        shadedColor = shadedColor * safe_rcp(ls.pdf) * float(world.numLights);
      }
    }
    else if (rendererState.renderMode == RenderMode::PrimitiveId)
      shadedColor = randomColor(result.primId).xyz();
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

    if (world.numLights > 0) {
      Ray &shadowRay = next.ray;
      shadowRay.ori = hitPos + sn * eps;
      shadowRay.dir = normalize(ls.dir);
      shadowRay.tmin = 0.f;
      shadowRay.tmax = ls.dist;//-1e-4f; // TODO: bias sample point
      shadowRay.time = ray.time;
      shadowRay.dbg = ray.dbg;
      next.rayType = Shadow;
      return;
    }

    // No shadow ray:
    throughput *= shadedColor;

    if (aoSamples < rendererState.ambientSamples) {
      vec3 u, v, w = sn;
      make_orthonormal_basis(u,v,w);
      auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
      vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

      Ray &aoRay = next.ray;
      aoRay.ori = hitPos + sn * eps;
      aoRay.dir = dir;
      aoRay.tmin = 0.f;
      aoRay.tmax = rendererState.occlusionDistance;
      aoRay.time = ray.time;
      aoRay.dbg = ray.dbg;
      next.rayType = AO;
      return;
    }

    // No AO:
    throughput += baseColor * rendererState.ambientColor * rendererState.ambientRadiance;

    return;
  } else if (rayType == Shadow) {
    int surfV = hr.hit ? 0 : 1;
    int volV = hitRec.volumeHit ? 0 : 1;

    float V = surfV * volV * hrv.Tr;
    throughput *= shadedColor * V;

    if (aoSamples < rendererState.ambientSamples) {
      vec3 u, v, w = sn;
      make_orthonormal_basis(u,v,w);
      auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
      vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

      Ray &aoRay = next.ray;
      aoRay.ori = hitPos + sn * eps;
      aoRay.dir = dir;
      aoRay.tmin = 0.f;
      aoRay.tmax = rendererState.occlusionDistance;
      aoRay.time = ray.time;
      aoRay.dbg = ray.dbg;
      next.rayType = AO;
      return;
    }

    // No AO:
    throughput += baseColor * rendererState.ambientColor * rendererState.ambientRadiance;

    return;
  } else if (rayType == AO) {
    aoSamples++;

    float weight = fmaxf(0.f, dot(ray.dir,sn));
    aoWeights += weight;
    if (weight > 0.f && hr.hit) {
      aoCount += weight;
    }

    if (aoSamples < rendererState.ambientSamples) {
      vec3 u, v, w = sn;
      make_orthonormal_basis(u,v,w);
      auto sp = cosine_sample_hemisphere(ss.random(), ss.random());
      vec3 dir = normalize(sp.x*u + sp.y*v + sp.z*w);

      Ray &aoRay = next.ray;
      aoRay.ori = hitPos + sn * eps;
      aoRay.dir = dir;
      aoRay.tmin = 0.f;
      aoRay.tmax = rendererState.occlusionDistance;
      aoRay.time = ray.time;
      aoRay.dbg = ray.dbg;
      next.rayType = AO;
      return;
    }

    float aoV = 0.f;
    if (aoWeights > 0.f) {
      aoV = 1.f - (aoCount/aoWeights);
    }

    throughput
        += baseColor * rendererState.ambientColor * rendererState.ambientRadiance * aoV;

    return;
  }
}

void VisionarayRendererDirectLight::renderFrame(const dco::Frame &frame,
                                                const dco::Camera &cam,
                                                uint2 size,
                                                VisionarayGlobalState *state,
                                                const DeviceObjectRegistry &DD,
                                                const RendererState &rendererState,
                                                unsigned worldID, int frameID)
{
  DevicePointer<DeviceObjectRegistry> onDevicePtr(&DD);
  DevicePointer<RendererState> rendererStatePtr(&rendererState);
  DevicePointer<dco::Frame> framePtr(&frame);
#ifdef WITH_CUDA
  cuda::for_each(state->renderingStream, 0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  hip::for_each(state->renderingStream, 0, size.x, 0, size.y,
#else
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const DeviceObjectRegistry &onDevice = *onDevicePtr;
        const auto &rendererState = *rendererStatePtr;
        const auto &frame = *framePtr;

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
            for (unsigned bounceID=0;true;++bounceID) {
              ray = clipRay(ray, rendererState.clipPlanes, rendererState.numClipPlanes);
              bool shadow = rayType == Shadow || rayType == AO;
              HitRec hitRec = intersectAll(ss, ray, worldID, onDevice, shadow);
              // 1. radiance
              // 2. shadow (optional)
              // 3. AO (optional)
              shade(ss, ray, rayType, worldID, onDevice,
                    rendererState,
                    hitRec,
                    shadeState,
                    ps);

              if (bounceID == 0) {
                firstHit = hitRec;
              }

              ray = shadeState.next.ray;
              rayType = shadeState.next.rayType;

              if (rayType == None) {
                break;
              }
            }

            if (firstHit.hit || shadeState.hdriMiss) {
              ps.color = float4(shadeState.throughput,1.f);
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
