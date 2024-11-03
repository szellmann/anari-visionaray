// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace visionaray {

Surface::Surface(VisionarayGlobalState *s) : Object(ANARI_SURFACE, s)
{
  vsurf = dco::createSurface();
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
  m_id = getParam<uint32_t>("id", ~0u);
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

uint32_t Surface::id() const
{
  return m_id;
}

Geometry *Surface::geometry()
{
  return m_geometry.ptr;
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
