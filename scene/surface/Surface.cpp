// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace visionaray {

Surface::Surface(VisionarayGlobalState *s) : Object(ANARI_SURFACE, s)
{
  vsurf.surfID = deviceState()->dcos.surfaces.alloc(vsurf);
  s->objectCounts.surfaces++;
}

Surface::~Surface()
{
  deviceState()->dcos.surfaces.free(vsurf.surfID);

  deviceState()->objectCounts.surfaces--;
}

void Surface::commit()
{
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");

  if (!m_material) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
    return;
  }

  if (!m_geometry) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
    return;
  }

  vsurf.geomID = m_geometry->visionarayGeometry().geomID;
  vsurf.matID = m_material->visionarayMaterial().matID;

  dispatch();
}

const Geometry *Surface::geometry() const
{
  return m_geometry.ptr;
}

const Material *Surface::material() const
{
  return m_material.ptr;
}

void Surface::dispatch()
{
  deviceState()->dcos.surfaces.update(vsurf.surfID, vsurf);

  // Upload/set accessible pointers
  deviceState()->onDevice.surfaces = deviceState()->dcos.surfaces.devicePtr();
}

// float4 Surface::getSurfaceColor(const Ray &ray) const
// {
//   auto &state = *deviceState();
//   auto &imc = state.invalidMaterialColor;
// 
//   auto *mat = material();
// 
//   if (!mat)
//     return float4(imc.x, imc.y, imc.z, 1.f);
// 
//   const auto colorAttribute = mat->colorAttribute();
//   const auto *colorSampler = mat->colorSampler();
//   if (colorSampler && colorSampler->isValid())
//     return colorSampler->getSample(*geometry(), ray);
//   else if (colorAttribute == Attribute::NONE)
//     return material()->color();
//   else
//     return geometry()->getAttributeValue(colorAttribute, ray);
// }
// 
// float Surface::getSurfaceOpacity(const Ray &ray) const
// {
//   auto &state = *deviceState();
//   auto &imc = state.invalidMaterialColor;
// 
//   auto *mat = material();
// 
//   if (!mat)
//     return 0.f;
// 
//   const auto opacityAttribute = mat->opacityAttribute();
//   const auto *opacitySampler = mat->opacitySampler();
//   if (opacitySampler && opacitySampler->isValid())
//     return opacitySampler->getSample(*geometry(), ray).x;
//   else if (opacityAttribute == Attribute::NONE)
//     return material()->opacity();
//   else
//     return geometry()->getAttributeValue(opacityAttribute, ray).x;
// }

void Surface::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest =
      helium::newTimeStamp();
}

bool Surface::isValid() const
{
  // bool allowInvalidMaterial = deviceState()->allowInvalidSurfaceMaterials;

  // if (allowInvalidMaterial) {
  //   return m_geometry && m_geometry->isValid();
  // } else {
    return m_geometry && m_material && m_geometry->isValid()
        && m_material->isValid();
  // }
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Surface *);
