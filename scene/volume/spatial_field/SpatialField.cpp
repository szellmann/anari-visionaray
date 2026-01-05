// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "SpatialField.h"
// subtypes
#include "BlockStructuredField.h"
#include "StructuredRegularField.h"
#include "UnstructuredField.h"
#ifdef WITH_NANOVDB
#include "NanoVDBField.h"
#endif

namespace visionaray {

SpatialField::SpatialField(VisionarayGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
    , m_gridAccel(s)
{
  vfield = dco::createSpatialField();
  vfield.fieldID = deviceState()->dcos.spatialFields.alloc(vfield);
}

SpatialField::~SpatialField()
{
  deviceState()->dcos.spatialFields.free(vfield.fieldID);
}

SpatialField *SpatialField::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "structuredRegular")
     return new StructuredRegularField(s);
  else if (subtype == "unstructured")
    return new UnstructuredField(s);
  else if (subtype == "amr" || subtype == "blockStructured")
    return new BlockStructuredField(s);
#ifdef WITH_NANOVDB
  else if (subtype == "nanovdb" || subtype == "vdb")
    return new NanoVDBField(s);
#endif
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

void SpatialField::setCellSize(float cellSize)
{
  vfield.cellSize = cellSize;
}

void SpatialField::dispatch()
{
  deviceState()->dcos.spatialFields.update(vfield.fieldID, vfield);

  // Upload/set accessible pointers
  deviceState()->onDevice.spatialFields = deviceState()->dcos.spatialFields.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::SpatialField *);
