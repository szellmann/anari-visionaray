// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "VolumeSampler.h"
#include "common.h"

namespace visionaray {

VolumeSampler::VolumeSampler(VisionarayGlobalState *s)
  : Sampler(s)
  , m_volume(this)
{
  vsampler.type = dco::Sampler::Volume;
}

bool VolumeSampler::isValid() const
{
  return m_volume && m_volume->isValid();
}

void VolumeSampler::commitParameters()
{
  Sampler::commitParameters();
  m_volume = getParamObject<Volume>("volume");
  m_inTransform = getParam<mat4>("inTransform", mat4::identity());
  m_inOffset = getParam<float4>("inOffset", float4(0.f, 0.f, 0.f, 0.f));
  m_outTransform = getParam<mat4>("outTransform", mat4::identity());
  m_outOffset = getParam<float4>("outOffset", float4(0.f, 0.f, 0.f, 0.f));
}

void VolumeSampler::finalize()
{
  vsampler.inTransform = m_inTransform;
  vsampler.inOffset = m_inOffset;
  vsampler.outTransform = m_outTransform;
  vsampler.outOffset = m_outOffset;
  vsampler.asVolume.volID = m_volume->visionarayVolume().volID;

  Sampler::dispatch();
}

} // namespace visionaray
