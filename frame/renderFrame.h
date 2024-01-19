#pragma once

#include "frame/common.h"
#include "frame/for_each.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

template <typename Rend>
inline void renderFrame(const dco::Frame &frame,
                        const dco::Camera &cam,
                        uint2 size,
                        Rend &rend,
                        VisionarayGlobalState *state,
                        const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
                        unsigned worldID,
                        int frameID,
                        int spp) {
#ifdef WITH_CUDA
  cuda::for_each(0, size.x, 0, size.y,
#else
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        ScreenSample ss{x, y, frameID, size, {/*RNG*/}};
        Ray ray;

        uint64_t clock_begin = clock64();

        if (rend.stochasticRendering) {
          // Need an RNG
          int pixelID = ss.x + ss.frameSize.x * ss.y;
          ss.random = Random(pixelID, frame.frameCounter);
        }

        float4 accumColor{0.f};
        PixelSample firstSample;
        for (int sampleID=0; sampleID<spp; ++sampleID) {

          float xf(x), yf(y);
          if constexpr(rend.stochasticRendering) {
            // jitter pixel sample
            vec2f jitter(ss.random() - .5f, ss.random() - .5f);
            xf += jitter.x;
            yf += jitter.y;
          }

          if (cam.type == dco::Camera::Pinhole)
            ray = cam.asPinholeCam.primary_ray(
                Ray{}, ss.random, xf, yf, float(size.x), float(size.y));
          else if (cam.type == dco::Camera::Matrix)
            ray = cam.asMatrixCam.primary_ray(
                Ray{}, xf, yf, float(size.x), float(size.y));
#if 1
          ray.dbg = ss.debug();
#endif

         PixelSample ps = rend.renderSample(ss,
                 ray,
                 worldID,
                 onDevice);
         accumColor += ps.color;
         if (sampleID == 0) {
           firstSample = ps;
         }
       }

       uint64_t clock_end = clock64();
       if (rend.rendererState.heatMapEnabled > 0.f) {
           float t = (clock_end - clock_begin)
               * (rend.rendererState.heatMapScale / spp);
           accumColor = over(vec4f(heatMap(t), .5f), accumColor);
       }

       // Color gets accumulated, depth, IDs, etc. are
       // taken from first sample
       PixelSample finalSample = firstSample;
       finalSample.color = accumColor*(1.f/spp);
       if constexpr(rend.supportsTaa)
         if (rend.rendererState.taaEnabled)
           frame.fillGBuffer(x, y, finalSample);
         else
           frame.writeSample(x, y, rend.rendererState.accumID, finalSample);
       else
         frame.writeSample(x, y, rend.rendererState.accumID, finalSample);
     });
}

} // namespace visionaray
