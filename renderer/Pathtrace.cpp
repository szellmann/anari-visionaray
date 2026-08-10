// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Pathtrace.h"

namespace visionaray {

Pathtrace::Pathtrace(VisionarayGlobalState *s) : Renderer(s)
{
  vrend.type = VisionarayRenderer::Pathtrace;
}

Pathtrace::~Pathtrace()
{}

void Pathtrace::commitParameters()
{
  Renderer::commitParameters();
  m_maxBounce = clamp(getParam<int>("maxBounce", 7), 0, 256);
  m_occlusionDistance = getParam<float>("ambientOcclusionDistance", 1e20f);
  m_ambientSamples = clamp(getParam<int>("ambientSamples", 1), 0, 256);
  m_pixelSamples = clamp(getParam<int>("pixelSamples", 1), 1, 256);
  m_sampleLimit = clamp(getParam<int>("sampleLimit", 1024), 1, INT_MAX);
  m_taaEnabled = getParam<bool>("taa", false);
  m_taaAlpha = getParam<float>("taaAlpha", 0.3f);
}

void Pathtrace::finalize()
{
  Renderer::finalize();
  vrend.rendererState.maxBounce = m_maxBounce;
  vrend.rendererState.occlusionDistance = m_occlusionDistance;
  vrend.rendererState.ambientSamples = m_ambientSamples;
  vrend.rendererState.pixelSamples = m_pixelSamples;
  vrend.rendererState.sampleLimit = m_sampleLimit;
  vrend.rendererState.taaEnabled = m_taaEnabled;
  vrend.rendererState.taaAlpha = m_taaAlpha;
}

} // namespace visionaray
