// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

// visionaray
#ifdef WITH_CUDA
#include <visionaray/texture/cuda_texture.h>
#elif defined(WITH_HIP)
#include <visionaray/texture/hip_texture.h>
#endif
// ours
#include "TransferFunction1D.h"

namespace visionaray {

TransferFunction1D::TransferFunction1D(VisionarayGlobalState *d)
  : Volume(d)
  , m_field(this)
  , m_colorData(this)
  , m_opacityData(this)
{
  vvol.type = dco::Volume::TransferFunction1D;
}

TransferFunction1D::~TransferFunction1D()
{
}

void TransferFunction1D::commitParameters()
{
  Volume::commitParameters();

  m_field = getParamObject<SpatialField>("value");
  if (!m_field) {
    // Some apps might still use "field" (from the provisional specs)
    m_field = getParamObject<SpatialField>("field");
  }

  m_valueRange = getParam<box1>("valueRange", box1(0.f, 1.f));

  m_colorData = getParamObject<Array1D>("color");
  m_opacityData = getParamObject<Array1D>("opacity");

  m_unitDistance = getParam<float>("unitDistance", 1.f);
}

void TransferFunction1D::finalize()
{
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  if (!m_field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid spatial field provided to transferFunction1D volume");
    return;
  }

  m_bounds = m_field->bounds();

  float4 constantColor{1.f};
  float constantOpacity{1.f};
  size_t numColorChannels{4};
  if (m_colorData) { // TODO: more types
    if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
      numColorChannels = 3;
  }

  float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
  float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

  size_t numColors = m_colorData ? m_colorData->size() : 1;
  size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
  size_t tfSize = max(numColors, numOpacities);

  std::vector<float4> tf(tfSize);
  for (size_t i=0; i<tfSize; ++i) {
    float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numColors-1) : 0.f;
    float colorFrac = colorPos-floorf(colorPos);

    float4 color0 = constantColor, color1 = constantColor;
    if (numColorChannels == 3) {
      float3 *colors = (float3 *)colorData;
      color0 = float4(colors[int(floorf(colorPos))], constantOpacity);
      color1 = float4(colors[int(ceilf(colorPos))], constantOpacity);
    }
    else if (numColorChannels == 4) {
      float4 *colors = (float4 *)colorData;
      color0 = colors[int(floorf(colorPos))];
      color1 = colors[int(ceilf(colorPos))];
    }

    float4 color = lerp_r(color0, color1, colorFrac);

    if (opacityData) {
      float alphaPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numOpacities-1) : 0.f;
      float alphaFrac = alphaPos-floorf(alphaPos);

      float alpha0 = opacityData[int(floorf(alphaPos))];
      float alpha1 = opacityData[int(ceilf(alphaPos))];

      color.w *= lerp_r(alpha0, alpha1, alphaFrac);
    }

    tf[i] = color;
  }
#if defined(WITH_CUDA) || defined(WITH_HIP)
  texture<float4, 1> tex(tf.size());
#else
  transFuncTexture = texture<float4, 1>(tf.size());
  auto &tex = transFuncTexture;
#endif
  tex.reset(tf.data());
  tex.set_filter_mode(Linear);
  tex.set_address_mode(Clamp);

  vvol.bounds = m_bounds;
  vvol.field = m_field->visionaraySpatialField();
  vvol.unitDistance = m_unitDistance;

  vvol.asTransferFunction1D.numValues = tex.size()[0];
  vvol.asTransferFunction1D.valueRange = m_valueRange;
#ifdef WITH_CUDA
  transFuncTexture.reset(tex);
  vvol.asTransferFunction1D.sampler = cuda_texture_ref<float4, 1>(transFuncTexture);
#elif defined(WITH_HIP)
  transFuncTexture = hip_texture<float4, 1>(tex);
  vvol.asTransferFunction1D.sampler = hip_texture_ref<float4, 1>(transFuncTexture);
#else
  vvol.asTransferFunction1D.sampler = texture_ref<float4, 1>(transFuncTexture);
#endif

  // Trigger a BVH rebuild:
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest = helium::newTimeStamp();

  dispatch();

  if (m_field->gridAccel().isValid())
    m_field->gridAccel().computeMaxOpacities(vvol.asTransferFunction1D);
}

void TransferFunction1D::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLSCommitSceneRequest =
      helium::newTimeStamp();
}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid();
}

aabb TransferFunction1D::bounds() const
{
  return m_bounds;
}

void TransferFunction1D::dispatch()
{
  deviceState()->dcos.volumes.update(vvol.volID, vvol);

  // Upload/set accessible pointers
  deviceState()->onDevice.volumes = deviceState()->dcos.volumes.devicePtr();
}

} // namespace visionaray
