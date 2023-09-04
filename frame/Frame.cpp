#include "Frame.h"
// std
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
// visionaray
#include "visionaray/detail/color_conversion.h"
#include "visionaray/detail/parallel_for.h"

namespace visionaray {

// Helper functions ///////////////////////////////////////////////////////////

static uint32_t cvt_uint32(const float &f)
{
  return static_cast<uint32_t>(255.f * std::clamp(f, 0.f, 1.f));
}

static uint32_t cvt_uint32(const float4 &v)
{
  return (cvt_uint32(v.x) << 0) | (cvt_uint32(v.y) << 8)
      | (cvt_uint32(v.z) << 16) | (cvt_uint32(v.w) << 24);
}

static uint32_t cvt_uint32_srgb(const float4 &v)
{
  return cvt_uint32(float4(linear_to_srgb(v.xyz()), v.w));
}

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
  s->objectCounts.frames++;
}

Frame::~Frame()
{
  wait();
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

  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);

  m_frameData.size = getParam<uint2>("size", uint2(10));
  m_frameData.invSize = 1.f / float2(m_frameData.size);

  const auto numPixels = m_frameData.size.x * m_frameData.size.y;

  m_perPixelBytes = 4 * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * m_perPixelBytes);

  m_depthBuffer.resize(m_depthType == ANARI_FLOAT32 ? numPixels : 0);
  m_accumBuffer.resize(numPixels, vec4{0.f});
  m_frameChanged = true;
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    if (flags & ANARI_WAIT)
      wait();
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

  auto start = std::chrono::steady_clock::now();

  state->commitBuffer.flush();

  if (!isValid()) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    std::fill(m_pixelBuffer.begin(), m_pixelBuffer.end(), 0);
    this->refDec(helium::RefType::INTERNAL);
    return;
  }

  if (state->commitBuffer.lastFlush() <= m_frameLastRendered) {
    if (!m_renderer->stochasticRendering()) {
      this->refDec(helium::RefType::INTERNAL);
      return;
    }
  }

  checkAccumulationReset();

  m_frameLastRendered = helium::newTimeStamp();
  state->currentFrame = this;

  m_future = async<void>([&, state, start]() {
    m_world->visionaraySceneUpdate();

    const auto &size = m_frameData.size;
    dco::Camera cam = m_camera->visionarayCamera();
    VisionarayRenderer &rend = m_renderer->visionarayRenderer();
    VisionarayScene scene = m_world->visionarayScene();

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.begin_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.begin_frame();

    if (m_nextFrameReset) {
      std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), vec4{0.f});
      rend.rendererState().accumID = 0;
      m_nextFrameReset = false;
    }

    parallel_for(state->threadPool,
        tiled_range2d<int>(0, size.x, 64, 0, size.y, 64),
        [&](range2d<int> r) {
          for (int y = r.cols().begin(); y != r.cols().end(); ++y) {
            for (int x = r.rows().begin(); x != r.rows().end(); ++x) {

              PRD prd{x, y, size, {/*RNG*/}};
              Ray ray;

              if (rend.stochasticRendering()) {
                // Need an RNG
                int pixelID = prd.x + prd.frameSize.x * prd.y;
                prd.random = Random(pixelID, rend.rendererState().accumID);
              }

              float4 accumColor{0.f};
              float depth;
              for (int sampleID=0; sampleID<rend.spp(); ++sampleID) {

                float xf(x), yf(y);
                if (rend.stochasticRendering()) {
                  // jitter pixel sample
                  vec2f jitter(prd.random() - .5f, prd.random() - .5f);
                  xf += jitter.x;
                  yf += jitter.y;
                }

                if (cam.type == dco::Camera::Pinhole)
                  ray = cam.asPinholeCam.primary_ray(
                      Ray{}, xf, yf, float(size.x), float(size.y));
                else if (cam.type == dco::Camera::Matrix)
                  ray = cam.asMatrixCam.primary_ray(
                      Ray{}, xf, yf, float(size.x), float(size.y));

                PixelSample ps = rend.renderSample(ray,
                        prd,
                        scene->m_worldID,
                        deviceState()->onDevice,
                        deviceState()->objectCounts);
                accumColor += ps.color;
                if (sampleID == 0) depth = ps.depth;
              }

              writeSample(x, y, {accumColor*(1.f/rend.spp()),depth});
            }
          }
        });

    if (cam.type == dco::Camera::Pinhole)
      cam.asPinholeCam.end_frame();
    else if (cam.type == dco::Camera::Matrix)
      cam.asMatrixCam.end_frame();

    rend.rendererState().accumID++;

    auto end = std::chrono::steady_clock::now();
    m_duration = std::chrono::duration<float>(end - start).count();
  });
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  *width = m_frameData.size.x;
  *height = m_frameData.size.y;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = m_colorType;
    return mapColorBuffer();
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return mapDepthBuffer();
  } else {
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
  return m_pixelBuffer.data();
}

void *Frame::mapDepthBuffer()
{
  return m_depthBuffer.data();
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

float2 Frame::screenFromPixel(const float2 &p) const
{
  return p * m_frameData.invSize;
}

void Frame::writeSample(int x, int y, PixelSample s)
{
  const auto idx = y * m_frameData.size.x + x;
  auto *color = m_pixelBuffer.data() + (idx * m_perPixelBytes);

  if (m_renderer->stochasticRendering()) {
    float alpha = 1.f / (m_renderer->visionarayRenderer().rendererState().accumID+1);
    m_accumBuffer[idx] = (1-alpha)*m_accumBuffer[idx] + alpha*s.color;
    s.color = m_accumBuffer[idx];
  }

  switch (m_colorType) {
  case ANARI_UFIXED8_VEC4: {
    auto c = cvt_uint32(s.color);
    std::memcpy(color, &c, sizeof(c));
    break;
  }
  case ANARI_UFIXED8_RGBA_SRGB: {
    auto c = cvt_uint32_srgb(s.color);
    std::memcpy(color, &c, sizeof(c));
    break;
  }
  case ANARI_FLOAT32_VEC4: {
    std::memcpy(color, &s.color, sizeof(s.color));
    break;
  }
  default:
    break;
  }
  if (!m_depthBuffer.empty())
    m_depthBuffer[idx] = s.depth;
}

void Frame::checkAccumulationReset()
{
  if (m_nextFrameReset)
    return;

  auto &state = *deviceState();
  if (m_lastCommitOccured < state.commitBuffer.lastFlush()) {
    m_lastCommitOccured = state.commitBuffer.lastFlush();
    m_nextFrameReset = true;
  }
  // if (m_lastUploadOccured < state.uploadBuffer.lastFlush()) {
  //   m_lastUploadOccured = state.uploadBuffer.lastFlush();
  //   m_nextFrameReset = true;
  // }
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Frame *);
