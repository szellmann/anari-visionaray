#pragma once

#include "camera/Camera.h"
#include "renderer/Renderer.h"
#include "scene/World.h"
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

  void dispatch();
  void detach();

  //// Data ////

  bool m_valid{false};

  dco::Frame vframe;

  std::vector<uint8_t> m_pixelBuffer;
  std::vector<float> m_depthBuffer;
  std::vector<float3> m_normalBuffer;
  std::vector<float3> m_albedoBuffer;
  std::vector<float4> m_motionVecBuffer;
  std::vector<uint32_t> m_primIdBuffer;
  std::vector<uint32_t> m_objIdBuffer;
  std::vector<uint32_t> m_instIdBuffer;
  std::vector<float4> m_accumBuffer;

  struct {
    bool enabled{false};
    std::vector<float4> currBuffer;
    std::vector<float4> prevBuffer;
    std::vector<float3> currAlbedoBuffer;
    std::vector<float3> prevAlbedoBuffer;
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
