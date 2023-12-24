#pragma once

#include "visionaray/texture/texture.h"
// ours
#include "camera/Camera.h"
#include "renderer/Renderer.h"
#include "scene/World.h"
#include "DeviceArray.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <future>
#include <vector>

namespace visionaray {

struct Frame : public helium::BaseFrame
{
  Frame(VisionarayGlobalState *s);
  ~Frame();

  bool isValid() const override;

  VisionarayGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

  void *mapColorBuffer();
  void *mapDepthBuffer();

  bool ready() const;
  void wait() const;

 private:
  void checkAccumulationReset();
  bool checkTAAReset();
  void mapBuffersOnDevice();

  void dispatch();

  //// Data ////

  bool m_valid{false};

  dco::Frame vframe;

  template <typename Array>
  void *mapHostDeviceArray(Array &arr)
  {
#ifdef WITH_CUDA
    if (1) {
      arr.unmapDevice();
      return (void *)arr.hostPtr();
    } else {
      return arr.devicePtr();
    }
#else
    return arr.devicePtr();
#endif
  }

  HostDeviceArray<uint8_t> m_pixelBuffer;
  HostDeviceArray<float> m_depthBuffer;
  HostDeviceArray<float3> m_normalBuffer;
  HostDeviceArray<float3> m_albedoBuffer;
  HostDeviceArray<float4> m_motionVecBuffer;
  HostDeviceArray<uint32_t> m_primIdBuffer;
  HostDeviceArray<uint32_t> m_objIdBuffer;
  HostDeviceArray<uint32_t> m_instIdBuffer;
  HostDeviceArray<float4> m_accumBuffer;

  struct {
    bool enabled{false};
    HostDeviceArray<float4> currBuffer;
    HostDeviceArray<float4> prevBuffer;
    HostDeviceArray<float3> currAlbedoBuffer;
    HostDeviceArray<float3> prevAlbedoBuffer;
    texture<float4, 2> history;
  } taa;

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  float m_duration{0.f};

  bool m_frameChanged{false};
  bool m_nextFrameReset{false};
  helium::TimeStamp m_cameraLastChanged{0};
  helium::TimeStamp m_rendererLastChanged{0};
  helium::TimeStamp m_worldLastChanged{0};
  helium::TimeStamp m_lastCommitOccured{0};
  helium::TimeStamp m_frameLastRendered{0};

  mutable std::future<void> m_future;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Frame *, ANARI_FRAME);
