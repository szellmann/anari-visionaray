// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
// subtypes
#include "Matte.h"

namespace visionaray {

Material::Material(VisionarayGlobalState *s) : Object(ANARI_MATERIAL, s)
{
  vmat.matID = s->objectCounts.materials++;
}

Material::~Material()
{
  detach();

  deviceState()->objectCounts.materials--;
}

Material *Material::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
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
  if (deviceState()->dcos.materials.size() <= vmat.matID) {
    deviceState()->dcos.materials.resize(vmat.matID+1);
  }
  deviceState()->dcos.materials[vmat.matID] = vmat;

  // Upload/set accessible pointers
  deviceState()->onDevice.materials = deviceState()->dcos.materials.data();
}

void Material::detach()
{
  if (deviceState()->dcos.materials.size() > vmat.matID) {
    if (deviceState()->dcos.materials[vmat.matID].matID == vmat.matID) {
      deviceState()->dcos.materials.erase(
          deviceState()->dcos.materials.begin() + vmat.matID);
    }
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.materials = deviceState()->dcos.materials.data();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Material *);
