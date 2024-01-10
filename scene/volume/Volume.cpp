// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

// visionaray
#ifdef WITH_CUDA
#include <visionaray/texture/cuda_texture.h> // include to prevent name collisions
#endif
// ours
#include "Volume.h"
// subtypes
#include "TransferFunction1D.h"

namespace visionaray {

Volume::Volume(VisionarayGlobalState *s) : Object(ANARI_VOLUME, s)
{
  memset(&vgeom,0,sizeof(vgeom));
  vgeom.type = dco::Geometry::Volume;
  vgeom.geomID = deviceState()->dcos.geometries.alloc(vgeom);
  s->objectCounts.volumes++;
}

Volume::~Volume()
{
  deviceState()->dcos.geometries.free(vgeom.geomID);

  deviceState()->objectCounts.volumes--;
}

Volume *Volume::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "transferFunction1D")
    return new TransferFunction1D(s);
  else
    return (Volume *)new UnknownObject(ANARI_VOLUME, s);
}

dco::Geometry Volume::visionarayGeometry() const
{
  return vgeom;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Volume *);
