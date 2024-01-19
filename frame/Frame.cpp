// std
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
// ours
#include "frame/common.h"
#include "scene/light/HDRI.h"
#include "Frame.h"
#include "for_each.h"
#include "renderFrame.h"

namespace visionaray {

template <typename R, typename TASK_T>
static std::future<R> async(TASK_T &&fcn)
{
  auto task = std::packaged_task<R()>(std::forward<TASK_T>(fcn));
  auto future = task.get_future();

  std::thread([task = std::move(task)]() mutable { task(); }).detach();

  return future;
}

template <typename R>
static bool is_ready(const std::future<R> &f)
{
  return !f.valid()
      || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Frame definitions //////////////////////////////////////////////////////////

Frame::Frame(VisionarayGlobalState *s) : helium::BaseFrame(s)
{
  vframe.frameID = deviceState()->dcos.frames.alloc(vframe);
  s->objectCounts.frames++;
}

Frame::~Frame()
{
  wait();
  deviceState()->dcos.frames.free(vframe.frameID);
  deviceState()->objectCounts.frames--;
}

bool Frame::isValid() const
{
  return m_valid;
}

VisionarayGlobalState *Frame::deviceState() const
{
  return (VisionarayGlobalState *)helium::BaseObject::m_state;
}

void Frame::commit()
{
  m_renderer = getParamObject<Renderer>("renderer");
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  m_camera = getParamObject<Camera>("camera");
  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  m_world = getParamObject<World>("world");
  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  m_valid = m_renderer && m_renderer->isValid() && m_camera
      && m_camera->isValid() && m_world && m_world->isValid();

  vframe.colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  vframe.depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  vframe.normalType = getParam<anari::DataType>("channel.normal", ANARI_UNKNOWN);
  vframe.albedoType = getParam<anari::DataType>("channel.albedo", ANARI_UNKNOWN);
  vframe.primIdType =
      getParam<anari::DataType>("channel.primitiveId", ANARI_UNKNOWN);
  vframe.objIdType = getParam<anari::DataType>("channel.objectId", ANARI_UNKNOWN);
  vframe.instIdType = getParam<anari::DataType>("channel.instanceId", ANARI_UNKNOWN);

  vframe.size = getParam<uint2>("size", uint2(10));
  vframe.invSize = 1.f / float2(vframe.size);

  const auto numPixels = vframe.size.x * vframe.size.y;

  vframe.stochasticRendering = m_renderer->stochasticRendering();

  vframe.perPixelBytes = 4 * (vframe.colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * vframe.perPixelBytes);

  m_depthBuffer.resize(vframe.depthType == ANARI_FLOAT32 ? numPixels : 0);
  m_accumBuffer.resize(numPixels, vec4{0.f});
  m_motionVecBuffer.resize(numPixels, vec4{0,0,0,1});
  m_frameChanged = true;

  m_normalBuffer.clear();
  m_albedoBuffer.clear();
  m_primIdBuffer.clear();
  m_objIdBuffer.clear();
  m_instIdBuffer.clear();

  if (vframe.normalType == ANARI_FLOAT32_VEC3)
    m_normalBuffer.resize(numPixels);
  if (vframe.albedoType == ANARI_FLOAT32_VEC3)
    m_albedoBuffer.resize(numPixels);
  if (vframe.primIdType == ANARI_UINT32)
    m_primIdBuffer.resize(numPixels);
  if (vframe.objIdType == ANARI_UINT32)
    m_objIdBuffer.resize(numPixels);
  if (vframe.instIdType == ANARI_UINT32)
    m_instIdBuffer.resize(numPixels);

  vframe.pixelBuffer = m_pixelBuffer.devicePtr();
  vframe.depthBuffer = m_depthBuffer.devicePtr();
  vframe.normalBuffer = m_normalBuffer.devicePtr();
  vframe.albedoBuffer = m_albedoBuffer.devicePtr();
  vframe.primIdBuffer = m_primIdBuffer.devicePtr();
  vframe.objIdBuffer = m_objIdBuffer.devicePtr();
  vframe.instIdBuffer = m_instIdBuffer.devicePtr();
  vframe.accumBuffer = m_accumBuffer.devicePtr();
  vframe.motionVecBuffer = m_motionVecBuffer.devicePtr();

  checkTAAReset();

  dispatch();
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }

  return 0;
}

void Frame::renderFrame()
{
  this->refInc(helium::RefType::INTERNAL);

  auto *state = deviceState();
  state->waitOnCurrentFrame();
  state->currentFrame = this;

  m_future = async<void>([&, state, this]() {
#ifdef WITH_CUDA
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
#else
    auto start = std::chrono::steady_clock::now();
#endif
    state->renderingSemaphore.frameStart();
    state->commitBufferFlush();

    if (!isValid()) {
      reportMessage(
          ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
#ifdef WITH_CUDA
      // TODO: outside this function!
#else
      std::fill(m_pixelBuffer.begin(), m_pixelBuffer.end(), 0);
#endif
      state->renderingSemaphore.frameEnd();
      return;
    }

    if (state->commitBufferLastFlush() <= m_frameLastRendered) {
      if (!m_renderer->stochasticRendering()) {
        state->renderingSemaphore.frameEnd();
        return;
      }
    }

    m_frameLastRendered = helium::newTimeStamp();

    checkAccumulationReset();
    // TAA is a parameter on the renderer; we check it here to
    // avoid having to use commit observers on the renderer
    if (checkTAAReset())
      dispatch();

    m_world->visionaraySceneUpdate();

    mapBuffersOnDevice();

    dco::Frame frame = this->vframe;
    const auto &size = frame.size;
    dco::Camera cam = m_camera->visionarayCamera();
    VisionarayRenderer &rend = m_renderer->visionarayRenderer();
    VisionarayScene scene = m_world->visionarayScene();

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.begin_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.begin_frame();

    if (m_nextFrameReset) {
#ifdef WITH_CUDA
      cuda::for_each(0, size.x, 0, size.y, [=] VSNRAY_GPU_FUNC (int x, int y) {
        frame.accumBuffer[x+size.x*y] = vec4{0.f};
      });
#else
      std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), vec4{0.f});
#endif
      rend.rendererState().accumID = 0;
      m_nextFrameReset = false;
    }

    rend.rendererState().envID = HDRI::backgroundID;

    if (cam.type == dco::Camera::Pinhole) {
      rend.rendererState().currMV = cam.asPinholeCam.get_view_matrix();
      rend.rendererState().currPR = cam.asPinholeCam.get_proj_matrix();
    } else if (cam.type == dco::Camera::Matrix) {
      rend.rendererState().currMV = cam.asMatrixCam.get_view_matrix();
      rend.rendererState().currPR = cam.asMatrixCam.get_proj_matrix();
    }

    int frameID = (int)vframe.frameCounter++; // modify the member here!
    auto worldID = scene->m_worldID;
    auto onDevice = state->onDevice;

    if (rend.type == VisionarayRenderer::Raycast)
      visionaray::renderFrame(
        frame, cam, size, rend.asRaycast.renderer, state, onDevice, worldID, frameID, rend.spp());
    else if (rend.type == VisionarayRenderer::DirectLight)
      visionaray::renderFrame(
        frame, cam, size, rend.asDirectLight.renderer, state, onDevice, worldID, frameID, rend.spp());
    else assert(0);

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.end_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.end_frame();

    rend.rendererState().prevMV = rend.rendererState().currMV;
    rend.rendererState().prevPR = rend.rendererState().currPR;

    rend.rendererState().accumID++;

    if (m_renderer->visionarayRenderer().taa()) {
      // Update history texture
      taa.history.reset(taa.prevBuffer.devicePtr());
#ifdef WITH_CUDA
#else
      frame.taa.history = texture_ref<float4, 2>(taa.history);
#endif

      // TAA pass
      parallel::for_each(state->threadPool, 0, size.x, 0, size.y,
          [=] VSNRAY_GPU_FUNC (int x, int y) {
            frame.toneMap(
                x, y, frame.accumSample(x, y, ~0, frame.pixelSample(x, y)));
          });

      // Copy buffers for next pass
      memcpy(taa.prevBuffer.devicePtr(), taa.currBuffer.devicePtr(),
          sizeof(taa.currBuffer[0]) * taa.currBuffer.size());
      memcpy(taa.prevAlbedoBuffer.devicePtr(), taa.currAlbedoBuffer.devicePtr(),
          sizeof(taa.currAlbedoBuffer[0]) * taa.currAlbedoBuffer.size());
    }

    state->renderingSemaphore.frameEnd();

#ifdef WITH_CUDA
    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
    m_duration = ms/1000.f;
#else
    auto end = std::chrono::steady_clock::now();
    m_duration = std::chrono::duration<float>(end - start).count();
#endif
  });
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  *width = vframe.size.x;
  *height = vframe.size.y;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = vframe.colorType;
    return mapColorBuffer();
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return mapDepthBuffer();
  } else if (channel == "channel.normal" && !m_normalBuffer.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return mapHostDeviceArray(m_normalBuffer);
  } else if (channel == "channel.albedo" && !m_albedoBuffer.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return mapHostDeviceArray(m_albedoBuffer);
  } else if (channel == "channel.primitiveId" && !m_primIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return mapHostDeviceArray(m_primIdBuffer);
  } else if (channel == "channel.objectId" && !m_objIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return mapHostDeviceArray(m_objIdBuffer);
  } else if (channel == "channel.instanceId" && !m_instIdBuffer.empty()) {
    *pixelType = ANARI_UINT32;
    return mapHostDeviceArray(m_instIdBuffer);
  }else {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }
}

void Frame::unmap(std::string_view channel)
{
  // no-op
}

int Frame::frameReady(ANARIWaitMask m)
{
  if (m == ANARI_NO_WAIT)
    return ready();
  else {
    wait();
    return 1;
  }
}

void Frame::discard()
{
  // no-op
}

void *Frame::mapColorBuffer()
{
  return mapHostDeviceArray(m_pixelBuffer);
}

void *Frame::mapDepthBuffer()
{
  return mapHostDeviceArray(m_depthBuffer);
}

bool Frame::ready() const
{
  return is_ready(m_future);
}

void Frame::wait() const
{
  if (m_future.valid()) {
    m_future.get();
    this->refDec(helium::RefType::INTERNAL);
    if (deviceState()->currentFrame == this)
      deviceState()->currentFrame = nullptr;
  }
}

void Frame::checkAccumulationReset()
{
  if (m_nextFrameReset)
    return;

  auto &state = *deviceState();
  if (m_lastCommitOccured < state.commitBufferLastFlush()) {
    m_lastCommitOccured = state.commitBufferLastFlush();
    m_nextFrameReset = true;
  }
  // if (m_lastUploadOccured < state.uploadBuffer.lastFlush()) {
  //   m_lastUploadOccured = state.uploadBuffer.lastFlush();
  //   m_nextFrameReset = true;
  // }
}

bool Frame::checkTAAReset()
{
  const auto numPixels = vframe.size.x * vframe.size.y;
  const float alpha = m_renderer->visionarayRenderer().rendererState().taaAlpha;

  const bool taaEnabled = m_renderer->visionarayRenderer().taa();

  const bool taaJustEnabled
      = m_renderer->visionarayRenderer().taa() && !vframe.taa.enabled;

  const bool taaJustDisabled
      = !m_renderer->visionarayRenderer().taa() && vframe.taa.enabled;

  if (taaJustEnabled || taaEnabled && numPixels != taa.currBuffer.size() ||
      taaEnabled && alpha != vframe.taa.alpha) {
    vframe.taa.enabled = true;
    vframe.taa.alpha = alpha;

    if (numPixels != taa.currBuffer.size()) {
      taa.currBuffer.resize(numPixels, vec4{0.f});
      taa.prevBuffer.resize(numPixels, vec4{0.f});
      taa.currAlbedoBuffer.resize(numPixels, vec3{0.f});
      taa.prevAlbedoBuffer.resize(numPixels, vec3{0.f});

      taa.history = texture<float4, 2>(vframe.size.x, vframe.size.y);
      taa.history.set_filter_mode(CardinalSpline);
      //taa.history.set_filter_mode(Nearest);
      //taa.history.set_filter_mode(Linear);
      taa.history.set_address_mode(Clamp);
      taa.history.set_normalized_coords(true);

      vframe.taa.currBuffer = taa.currBuffer.devicePtr();
      vframe.taa.prevBuffer = taa.prevBuffer.devicePtr();
      vframe.taa.currAlbedoBuffer = taa.currAlbedoBuffer.devicePtr();
      vframe.taa.prevAlbedoBuffer = taa.prevAlbedoBuffer.devicePtr();
    }
    return true;
  } else if (taaJustDisabled) {
    vframe.taa.enabled = false;
    return  true;
  } else {
    return false;
  }
}

void Frame::mapBuffersOnDevice()
{
  vframe.pixelBuffer  = (uint8_t *)m_pixelBuffer.mapDevice();
  vframe.depthBuffer  = (float *)m_depthBuffer.mapDevice();
  vframe.normalBuffer = (float3 *)m_normalBuffer.mapDevice();
  vframe.albedoBuffer = (float3 *)m_albedoBuffer.mapDevice();
  vframe.primIdBuffer = (uint32_t *)m_primIdBuffer.mapDevice();
  vframe.objIdBuffer  = (uint32_t *)m_objIdBuffer.mapDevice();
  vframe.instIdBuffer = (uint32_t *)m_instIdBuffer.mapDevice();
}

void Frame::dispatch()
{
  deviceState()->dcos.frames.update(vframe.frameID, vframe);

  // Upload/set accessible pointers
  deviceState()->onDevice.frames = deviceState()->dcos.frames.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Frame *);
