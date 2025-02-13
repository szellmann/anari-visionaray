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

void Renderer::commitParameters()
{
  m_clipPlanes = getParamObject<Array1D>("clipPlane");
  m_bgColor = getParam<float4>("background", float4(float3(0.f), 1.f));
  m_ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
  m_ambientRadiance = getParam<float>("ambientRadiance", 0.2f);
  m_renderMode = getParamString("mode", "default");
  m_heatMapEnabled = getParam<bool>("heatMapEnabled", false);
  m_heatMapScale = getParam<float>("heatMapScale", 0.1f);
  m_taaEnabled = getParam<bool>("taa", false);
  m_taaAlpha = getParam<float>("taaAlpha", 0.3f);
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

  vrend.rendererState.bgColor = m_bgColor;
  vrend.rendererState.ambientColor = m_ambientColor;
  vrend.rendererState.ambientRadiance = m_ambientRadiance;
  if (m_renderMode == "default")
    vrend.rendererState.renderMode = RenderMode::Default;
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
  vrend.rendererState.heatMapEnabled = m_heatMapEnabled;
  vrend.rendererState.heatMapScale = m_heatMapScale;
  vrend.rendererState.taaEnabled = m_taaEnabled;
  vrend.rendererState.taaAlpha = m_taaAlpha;
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
