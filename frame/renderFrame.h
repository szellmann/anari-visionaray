#pragma once

#include "frame/common.h"
#include "frame/for_each.h"
#include "renderer/Renderer.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

void renderFrame_Raycast(const dco::Frame &frame,
                         const dco::Camera &cam,
                         const VisionarayRenderer &rend,
                         uint2 size,
                         VisionarayGlobalState *state,
                         const VisionarayGlobalState::DeviceObjectRegistry &DD,
                         unsigned worldID, int frameID, int spp);

void renderFrame_DirectLight(const dco::Frame &frame,
                             const dco::Camera &cam,
                             const VisionarayRenderer &rend,
                             uint2 size,
                             VisionarayGlobalState *state,
                             const VisionarayGlobalState::DeviceObjectRegistry &DD,
                             unsigned worldID, int frameID, int spp);

template <typename Rend>
inline void renderFrame(const dco::Frame &frame,
                        const dco::Camera &cam,
                        const Rend rend,
                        uint2 size,
                        VisionarayGlobalState *state,
                        const VisionarayGlobalState::DeviceObjectRegistry &DD,
                        unsigned worldID, int frameID, int spp) {
#ifdef WITH_CUDA
  VisionarayGlobalState::DeviceObjectRegistry *onDevicePtr;
  CUDA_SAFE_CALL(cudaMalloc(&onDevicePtr, sizeof(DD)));
  CUDA_SAFE_CALL(cudaMemcpy(onDevicePtr, &DD, sizeof(DD), cudaMemcpyHostToDevice));
  cuda::for_each(0, size.x, 0, size.y,
#else
  auto *onDevicePtr = &DD;
  parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
#endif
      [=] VSNRAY_GPU_FUNC (int x, int y) {

        const VisionarayGlobalState::DeviceObjectRegistry &onDevice = *onDevicePtr;

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
          else if (cam.type == dco::Camera::Ortho)
            ray = cam.asOrthoCam.primary_ray(
                Ray{}, xf, yf, float(size.x), float(size.y));
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
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(onDevicePtr));
#endif
}

} // namespace visionaray
