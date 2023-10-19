// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Raycast.h"
#include "DirectLight.h"
#include "Renderer.h"

namespace visionaray {

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(VisionarayGlobalState *s) : Object(ANARI_RENDERER, s)
{
  s->objectCounts.renderers++;

  Array1DMemoryDescriptor md;
  md.elementType = ANARI_FLOAT32_VEC3;
  md.numItems = 4;
}

Renderer::~Renderer()
{
  deviceState()->objectCounts.renderers--;
}

void Renderer::commit()
{
  // variables supported by ALL renderers
  auto commitCommonState = [this](auto &state) {
    state.bgColor = getParam<float4>("background", float4(float3(0.f), 1.f));
    std::string renderMode = getParamString("mode", "default");
    if (renderMode == "default")
      state.renderMode = RenderMode::Default;
    else if (renderMode == "Ng")
      state.renderMode = RenderMode::Ng;
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
  };

  if (vrend.type == VisionarayRenderer::Raycast) {
    auto &renderState = vrend.asRaycast.renderer.rendererState;
    commitCommonState(renderState);
  } else if (vrend.type == VisionarayRenderer::DirectLight) {
    auto &renderState = vrend.asDirectLight.renderer.rendererState;
    commitCommonState(renderState);
    renderState.ambientColor = getParam<vec3>("ambientColor", vec3(1.f));
    renderState.ambientRadiance = getParam<float>("ambientRadiance", 0.f);
    renderState.occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
    renderState.ambientSamples = clamp(getParam<int>("ambientSamples", 1), 0, 256);
    renderState.pixelSamples = clamp(getParam<int>("pixelSamples", 1), 1, 256);
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
