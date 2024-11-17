// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Raycast.h"
#include "DirectLight.h"
#include "Renderer.h"

namespace visionaray {

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(VisionarayGlobalState *s)
  : Object(ANARI_RENDERER, s)
  , m_clipPlanes(this)
{
}

void Renderer::commit()
{
  // variables supported by ALL renderers
  auto commitCommonState = [this](auto &state) {
    m_clipPlanes = getParamObject<Array1D>("clipPlane");
    if (m_clipPlanes) {
      m_clipPlanesOnDevice.resize(m_clipPlanes->size());
      for (size_t i=0; i<m_clipPlanes->size(); ++i) {
        m_clipPlanesOnDevice[i] = m_clipPlanes->beginAs<float4>()[i];
      }
      state.clipPlanes = m_clipPlanesOnDevice.devicePtr();
      state.numClipPlanes = (unsigned)m_clipPlanesOnDevice.size();
    } else {
      state.clipPlanes = nullptr;
      state.numClipPlanes = 0;
    }

    state.bgColor = getParam<float4>("background", float4(float3(0.f), 1.f));
    state.ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
    state.ambientRadiance = getParam<float>("ambientRadiance", 0.2f);
    std::string renderMode = getParamString("mode", "default");
    if (renderMode == "default")
      state.renderMode = RenderMode::Default;
    else if (renderMode == "Ng")
      state.renderMode = RenderMode::Ng;
    else if (renderMode == "Ns")
      state.renderMode = RenderMode::Ns;
    else if (renderMode == "tangent")
      state.renderMode = RenderMode::Tangent;
    else if (renderMode == "bitangent")
      state.renderMode = RenderMode::Bitangent;
    else if (renderMode == "albedo")
      state.renderMode = RenderMode::Albedo;
    else if (renderMode == "motionVec")
      state.renderMode = RenderMode::MotionVec;
    else if (renderMode == "geometry.attribute0")
      state.renderMode = RenderMode::GeometryAttribute0;
    else if (renderMode == "geometry.attribute1")
      state.renderMode = RenderMode::GeometryAttribute1;
    else if (renderMode == "geometry.attribute2")
      state.renderMode = RenderMode::GeometryAttribute2;
    else if (renderMode == "geometry.attribute3")
      state.renderMode = RenderMode::GeometryAttribute3;
    else if (renderMode == "geometry.color")
      state.renderMode = RenderMode::GeometryColor;
    state.heatMapEnabled = getParam<bool>("heatMapEnabled", false);
    state.heatMapScale = getParam<float>("heatMapScale", 0.1f);
    state.taaEnabled = getParam<bool>("taa", false);
    state.taaAlpha = getParam<float>("taaAlpha", 0.3f);
  };

  commitCommonState(vrend.rendererState);
  if (vrend.type == VisionarayRenderer::DirectLight) {
    vrend.rendererState.occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
    vrend.rendererState.ambientSamples = clamp(getParam<int>("ambientSamples", 1), 0, 256);
    vrend.rendererState.pixelSamples = clamp(getParam<int>("pixelSamples", 1), 1, 256);
  }
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
