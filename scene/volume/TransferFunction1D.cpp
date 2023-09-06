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

  m_bounds = m_field ? m_field->bounds() : aabb();

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

  std::vector<float4> tf(m_colorData->size());
  for (size_t i=0; i<tf.size(); ++i) {
    tf[i] = vec4(colorData[i],opacityData[i]);
  }
  transFuncTexture = texture<float4, 1>(tf.size());
  transFuncTexture.reset(tf.data());
  transFuncTexture.set_filter_mode(Linear);
  transFuncTexture.set_address_mode(Clamp);

  vgeom.asVolume.data.bounds = m_bounds;

  dispatch();
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
