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
}

SpatialField::~SpatialField()
{
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
  if (deviceState()->dcos.spatialFields.size() <= vfield.fieldID) {
    deviceState()->dcos.spatialFields.resize(vfield.fieldID+1);
  }
  deviceState()->dcos.spatialFields[vfield.fieldID] = vfield;

  // Upload/set accessible pointers
  deviceState()->onDevice.spatialFields = deviceState()->dcos.spatialFields.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::SpatialField *);
