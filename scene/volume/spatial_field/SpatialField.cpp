// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "SpatialField.h"
// subtypes
#include "StructuredRegularField.h"
#include "UnstructuredField.h"

namespace visionaray {

SpatialField::SpatialField(VisionarayGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
{
  vfield.fieldID = s->objectCounts.spatialFields++;
  m_gridAccel.visionarayAccel().fieldID = vfield.fieldID;
}

SpatialField::~SpatialField()
{
  detach();

  m_gridAccel.cleanup();

  deviceState()->objectCounts.spatialFields--;
}

SpatialField *SpatialField::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
   if (subtype == "structuredRegular")
     return new StructuredRegularField(s);
  else if (subtype == "unstructured")
    return new UnstructuredField(s);
  else
    return (SpatialField *)new UnknownObject(ANARI_SPATIAL_FIELD, s);
}

dco::SpatialField SpatialField::visionaraySpatialField() const
{
  return vfield;
}

GridAccel &SpatialField::gridAccel()
{
  return m_gridAccel;
}

void SpatialField::buildGrid()
{
  reportMessage(ANARI_SEVERITY_WARNING,
      "buildGrid() not implemented for field type");
}

float SpatialField::stepSize() const
{
  return vfield.baseDT;
}

void SpatialField::setStepSize(float size)
{
  vfield.baseDT = size;
}

void SpatialField::dispatch()
{
  m_gridAccel.dispatch(deviceState());

  if (deviceState()->dcos.spatialFields.size() <= vfield.fieldID) {
    deviceState()->dcos.spatialFields.resize(vfield.fieldID+1);
  }
  deviceState()->dcos.spatialFields[vfield.fieldID] = vfield;

  // Upload/set accessible pointers
  deviceState()->onDevice.spatialFields = deviceState()->dcos.spatialFields.data();
}

void SpatialField::detach()
{
  m_gridAccel.detach(deviceState());

  if (deviceState()->dcos.spatialFields.size() > vfield.fieldID) {
    if (deviceState()->dcos.spatialFields[vfield.fieldID].fieldID == vfield.fieldID) {
      deviceState()->dcos.spatialFields.erase(
          deviceState()->dcos.spatialFields.begin() + vfield.fieldID);
    }
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.spatialFields = deviceState()->dcos.spatialFields.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::SpatialField *);
