// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "TransferFunction1D.h"

namespace visionaray {

TransferFunction1D::TransferFunction1D(VisionarayGlobalState *d) : Volume(d)
{
  vgeom.type = dco::Geometry::Volume;
  vgeom.asVolume.data.type = dco::Volume::TransferFunction1D;
}

TransferFunction1D::~TransferFunction1D()
{
  detach();
}

void TransferFunction1D::commit()
{
  m_field = getParamObject<SpatialField>("field");
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  m_bounds = m_field->bounds();

  m_valueRange = getParam<box1>("valueRange", box1(0.f, 1.f));

  m_colorData = getParamObject<Array1D>("color");
  m_opacityData = getParamObject<Array1D>("opacity");
  m_densityScale = getParam<float>("densityScale", 1.f);

  if (!m_colorData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no color data provided to transferFunction1D volume");
    return;
  }

  if (!m_opacityData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no opacity data provided to transfer function");
    return;
  }

  auto *colorData = m_colorData->beginAs<vec3>();
  auto *opacityData = m_opacityData->beginAs<float>();

  size_t tfSize = max(m_colorData->size(), m_opacityData->size());

  std::vector<float4> tf(tfSize);
  for (size_t i=0; i<tfSize; ++i) {
    float colorPos = (float(i)/(tfSize-1))*(m_colorData->size()-1);
    float colorFrac = colorPos-floorf(colorPos);

    vec3f color0 = colorData[int(floorf(colorPos))];
    vec3f color1 = colorData[int(ceilf(colorPos))];
    vec3f color = lerp(color0, color1, colorFrac);

    float alphaPos = (float(i)/(tfSize-1))*(m_opacityData->size()-1);
    float alphaFrac = alphaPos-floorf(alphaPos);

    float alpha0 = opacityData[int(floorf(alphaPos))];
    float alpha1 = opacityData[int(ceilf(alphaPos))];
    float alpha = lerp(alpha0, alpha1, alphaFrac);

    tf[i] = vec4(color, alpha);
  }
  transFuncTexture = texture<float4, 1>(tf.size());
  transFuncTexture.reset(tf.data());
  transFuncTexture.set_filter_mode(Linear);
  transFuncTexture.set_address_mode(Clamp);

  vgeom.asVolume.data.bounds = m_bounds;
  vgeom.asVolume.data.volID = m_field->visionaraySpatialField().fieldID;
  vgeom.asVolume.data.fieldID = m_field->visionaraySpatialField().fieldID;

  dispatch();

  m_field->gridAccel().computeMaxOpacities(
      deviceState()->onDevice.transferFunctions[vgeom.asVolume.data.volID]);
}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid() && m_colorData && m_opacityData;
}

aabb TransferFunction1D::bounds() const
{
  return m_bounds;
}

void TransferFunction1D::dispatch()
{
  if (deviceState()->dcos.transferFunctions.size() <= vgeom.asVolume.data.volID) {
    deviceState()->dcos.transferFunctions.resize(vgeom.asVolume.data.volID+1);
  }
  deviceState()->dcos.transferFunctions[vgeom.asVolume.data.volID].volID
      = vgeom.asVolume.data.volID;
  deviceState()->dcos.transferFunctions[vgeom.asVolume.data.volID].as1D.numValues
      = transFuncTexture.size()[0];
  deviceState()->dcos.transferFunctions[vgeom.asVolume.data.volID].as1D.valueRange
      = m_valueRange;
  deviceState()->dcos.transferFunctions[vgeom.asVolume.data.volID].as1D.sampler
      = texture_ref<float4, 1>(transFuncTexture);

  // Upload/set accessible pointers
  deviceState()->onDevice.transferFunctions
      = deviceState()->dcos.transferFunctions.data();
}

void TransferFunction1D::detach()
{
  if (deviceState()->dcos.transferFunctions.size() > vgeom.asVolume.data.volID) {
    if (deviceState()->dcos.transferFunctions[vgeom.asVolume.data.volID].volID
        == vgeom.asVolume.data.volID) {
      deviceState()->dcos.transferFunctions.erase(
          deviceState()->dcos.transferFunctions.begin() + vgeom.asVolume.data.volID);
    }
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.transferFunctions
      = deviceState()->dcos.transferFunctions.data();
}

} // namespace visionaray
