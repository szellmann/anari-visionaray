// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
// subtypes
#include "Matte.h"
#include "PBM.h"

namespace visionaray {

Material::Material(VisionarayGlobalState *s) : Object(ANARI_MATERIAL, s)
{
  vmat = dco::createMaterial();
  vmat.matID = deviceState()->dcos.materials.alloc(vmat);
  s->objectCounts.materials++;
}

Material::~Material()
{
  deviceState()->dcos.materials.free(vmat.matID);
  deviceState()->objectCounts.materials--;
}

Material *Material::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
  else if (subtype == "physicallyBased")
    return new PBM(s);
  else
    return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

void Material::commit()
{
  // m_alphaMode = alphaModeFromString(getParamString("alphaMode", "opaque"));
  // m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
}

void Material::dispatch()
{
  deviceState()->dcos.materials.update(vmat.matID, vmat);

  // Upload/set accessible pointers
  deviceState()->onDevice.materials = deviceState()->dcos.materials.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Material *);
