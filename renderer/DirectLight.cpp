// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "DirectLight.h"

namespace visionaray {

DirectLight::DirectLight(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::DirectLight;
}

DirectLight::~DirectLight()
{}

void DirectLight::commitParameters()
{
  Renderer::commitParameters();
  m_occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
  m_ambientSamples = clamp(getParam<int>("ambientSamples", 1), 0, 256);
  m_pixelSamples = clamp(getParam<int>("pixelSamples", 1), 1, 256);
  m_taaEnabled = getParam<bool>("taa", false);
  m_taaAlpha = getParam<float>("taaAlpha", 0.3f);
}

void DirectLight::finalize()
{
  Renderer::finalize();
  vrend.rendererState.occlusionDistance = m_occlusionDistance;
  vrend.rendererState.ambientSamples = m_ambientSamples;
  vrend.rendererState.pixelSamples = m_pixelSamples;
  vrend.rendererState.taaEnabled = m_taaEnabled;
  vrend.rendererState.taaAlpha = m_taaAlpha;
}

} // namespace visionaray
