
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

  auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID], /*shadow:*/false);

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
          mat, onDevice, attribs, localHitPos, hr.prim_id, tng, btng, sn);
    }
    vec4f color = getColor(mat, onDevice, attribs, localHitPos, hr.prim_id);

    // That doesn't work for instances..
    float3 shadedColor{0.f};

    if (rendererState.renderMode == RenderMode::Default) {
      float3 viewDir = -ray.dir;
      auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
      for (unsigned lightID=0; lightID<world.numLights; ++lightID) {
        const dco::Light &light = getLight(world.allLights, lightID, onDevice);

        LightSample ls = sampleLight(light, hitPos, ss.random);

        float3 brdf = evalMaterial(mat,
                                   onDevice,
                                   attribs,
                                   localHitPos,
                                   hr.prim_id,
                                   gn, sn,
                                   normalize(viewDir),
                                   normalize(ls.dir),
                                   ls.intensity * safe_rcp(ls.dist2));
        shadedColor += brdf * safe_rcp(ls.pdf);
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
      shadedColor = (tng + float3(1.f)) * float3(0.5f);
    else if (rendererState.renderMode == RenderMode::Bitangent)
      shadedColor = (btng + float3(1.f)) * float3(0.5f);
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


    float a = getOpacity(mat, onDevice, attribs, localHitPos, hr.prim_id);
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
      hitPos = ray.ori + hr.t * ray.dir;
      const float eps = epsilonFrom(hitPos, ray.dir, hr.t);
      ray.tmin = hr.t + eps;
      hr = intersectSurfaces(ray, onDevice.TLSs[worldID], /*shadow:*/false);
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
    const dco::Volume &vol = onDevice.volumes[group.volumes[hrv.localID]];

    float3 color(0.f);
    float alpha = 0.f;

    mat4 invXfm = inverse(inst.xfms[0]);

    Ray localRay = ray;
    localRay.ori = (invXfm * float4(ray.ori, 1.f)).xyz();
    localRay.dir = (invXfm * float4(ray.dir, 0.f)).xyz();

    if (rendererState.gradientShading) {
      rayMarchVolume<1>(ss,
                        onDevice,
                        localRay,
                        vol,
                        world.allLights,
                        world.numLights,
                        rendererState.ambientColor,
                        rendererState.ambientRadiance,
                        rendererState.volumeSamplingRateInv,
                        color,
                        alpha);
    } else {
      rayMarchVolume<0>(ss,
                        onDevice,
                        localRay,
                        vol,
                        world.allLights,
                        world.numLights,
                        rendererState.ambientColor,
                        rendererState.ambientRadiance,
                        rendererState.volumeSamplingRateInv,
                        color,
                        alpha);
    }

    result.color = over(float4(color,alpha), result.color);
    result.Ng = float3{}; // TODO: gradient
    result.Ns = float3{}; // TODO..
    result.albedo = float3{}; // TODO..
    result.objId = group.objIds[hrv.localID];
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
  DevicePointer<DeviceObjectRegistry> onDevicePtr(&DD);
  DevicePointer<RendererState> rendererStatePtr(&rendererState);
  DevicePointer<dco::Frame> framePtr(&frame);
#ifdef WITH_CUDA
  cuda::for_each(0, size.x, 0, size.y,
#elif defined(WITH_HIP)
  hip::for_each(0, size.x, 0, size.y,
#else
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
}

} // namespace visionaray
