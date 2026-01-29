// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Raycast.h"
#include "DirectLight.h"
#include "Renderer.h"
#include "scene/surface/material/sampler/common.h" // for imageSamplerUpdateData (TODO!)

namespace visionaray {

// VisionarayRenderer impl  ///////////////////////////////////////////////////

void VisionarayRenderer::renderFrame(const dco::Frame &frame,
                                     const dco::Camera &cam,
                                     uint2 size,
                                     VisionarayGlobalState *state,
                                     const DeviceObjectRegistry &DD,
                                     unsigned worldID, int frameID)
{
  DevicePointer<DeviceObjectRegistry> onDevicePtr(&DD);
  DevicePointer<RendererState> rendererStatePtr(&rendererState);
  DevicePointer<dco::Frame> framePtr(&frame);
  DevicePointer<dco::Camera> camPtr(&cam);
  if (type == Raycast) {
    asRaycast.renderFrame(
        onDevicePtr, rendererStatePtr, framePtr, camPtr, size, state, worldID, frameID);
  } else if (type == DirectLight) {
    asDirectLight.renderFrame(
        onDevicePtr, rendererStatePtr, framePtr, camPtr, size, state, worldID, frameID);
  }
}

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(VisionarayGlobalState *s)
  : Object(ANARI_RENDERER, s)
  , m_clipPlanes(this)
{
}

void Renderer::commitParameters()
{
  m_clipPlanes = getParamObject<Array1D>("clipPlane");
  m_bgImage = getParamObject<Array2D>("background");
  m_bgColor = getParam<float4>("background", float4(float3(0.f), 1.f));
  m_ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
  m_ambientRadiance = getParam<float>("ambientRadiance", 0.2f);
  m_renderMode = getParamString("mode", "default");
  m_gradientShading = getParam<bool>("gradientShading", false);
  m_heatMapEnabled = getParam<bool>("heatMapEnabled", false);
  m_heatMapScale = getParam<float>("heatMapScale", 0.1f);
}

void Renderer::finalize()
{
  m_clipPlanes = getParamObject<Array1D>("clipPlane");
  if (m_clipPlanes) {
    m_clipPlanesOnDevice.resize(m_clipPlanes->size());
    for (size_t i=0; i<m_clipPlanes->size(); ++i) {
      m_clipPlanesOnDevice[i] = m_clipPlanes->beginAs<float4>()[i];
    }
    vrend.rendererState.clipPlanes = m_clipPlanesOnDevice.devicePtr();
    vrend.rendererState.numClipPlanes = (unsigned)m_clipPlanesOnDevice.size();
  } else {
    vrend.rendererState.clipPlanes = nullptr;
    vrend.rendererState.numClipPlanes = 0;
  }

  memset(&vrend.rendererState.bgImage, 0, sizeof(vrend.rendererState.bgImage));

  if (m_bgImage) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    texture<vector<4, unorm<8>>, 2> tex(m_bgImage->size().x, m_bgImage->size().y);
#else
    m_bgTexture
        = texture<vector<4, unorm<8>>, 2>(m_bgImage->size().x, m_bgImage->size().y);
    auto &tex = m_bgTexture;
#endif

    if (!imageSamplerUpdateData(tex, m_bgImage)) { // TODO: move this function upwards
      reportMessage(ANARI_SEVERITY_WARNING,
          "unsupported element type for background image: %s",
          anari::toString(m_bgImage->elementType()));
      return;
    }

    tex.set_filter_mode(Linear);
    tex.set_address_mode(0, Clamp);
    tex.set_address_mode(1, Clamp);

#ifdef WITH_CUDA
    m_bgTexture = cuda_texture<vector<4, unorm<8>>, 2>(tex);
#elif defined(WITH_HIP)
    m_bgTexture = hip_texture<vector<4, unorm<8>>, 2>(tex);
#endif

#ifdef WITH_CUDA
  vrend.rendererState.bgImage = cuda_texture_ref<vector<4, unorm<8>>, 2>(m_bgTexture);
#elif defined(WITH_HIP)
  vrend.rendererState.bgImage = hip_texture_ref<vector<4, unorm<8>>, 2>(m_bgTexture);
#else
  vrend.rendererState.bgImage = texture_ref<vector<4, unorm<8>>, 2>(m_bgTexture);
#endif
  }

  vrend.rendererState.bgColor = m_bgColor;
  vrend.rendererState.ambientColor = m_ambientColor;
  vrend.rendererState.ambientRadiance = m_ambientRadiance;
  if (m_renderMode == "default")
    vrend.rendererState.renderMode = RenderMode::Default;
  else if (m_renderMode == "primitiveId")
    vrend.rendererState.renderMode = RenderMode::PrimitiveId;
  else if (m_renderMode == "worldPosition")
    vrend.rendererState.renderMode = RenderMode::WorldPosition;
  else if (m_renderMode == "objectPosition")
    vrend.rendererState.renderMode = RenderMode::ObjectPosition;
  else if (m_renderMode == "Ng")
    vrend.rendererState.renderMode = RenderMode::Ng;
  else if (m_renderMode == "Ns")
    vrend.rendererState.renderMode = RenderMode::Ns;
  else if (m_renderMode == "tangent")
    vrend.rendererState.renderMode = RenderMode::Tangent;
  else if (m_renderMode == "bitangent")
    vrend.rendererState.renderMode = RenderMode::Bitangent;
  else if (m_renderMode == "albedo")
    vrend.rendererState.renderMode = RenderMode::Albedo;
  else if (m_renderMode == "motionVec")
    vrend.rendererState.renderMode = RenderMode::MotionVec;
  else if (m_renderMode == "geometry.attribute0")
    vrend.rendererState.renderMode = RenderMode::GeometryAttribute0;
  else if (m_renderMode == "geometry.attribute1")
    vrend.rendererState.renderMode = RenderMode::GeometryAttribute1;
  else if (m_renderMode == "geometry.attribute2")
    vrend.rendererState.renderMode = RenderMode::GeometryAttribute2;
  else if (m_renderMode == "geometry.attribute3")
    vrend.rendererState.renderMode = RenderMode::GeometryAttribute3;
  else if (m_renderMode == "geometry.color")
    vrend.rendererState.renderMode = RenderMode::GeometryColor;
  vrend.rendererState.gradientShading = m_gradientShading;
  vrend.rendererState.heatMapEnabled = m_heatMapEnabled;
  vrend.rendererState.heatMapScale = m_heatMapScale;
}

Renderer *Renderer::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "direct_light" || subtype == "default")
    return new DirectLight(s);
  else if (subtype == "raycast")
    return new Raycast(s);
  else
    return {};
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Renderer *);
