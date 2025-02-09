// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
// subtypes
#include "BezierCurve.h"
#include "Cone.h"
//#include "Curve.h"
#include "Cylinder.h"
#include "ISOSurface.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"
// std
#include <cstring>
#include <limits>

namespace visionaray {

Geometry::Geometry(VisionarayGlobalState *s) : Object(ANARI_GEOMETRY, s)
{
  vgeom = dco::createGeometry();
  vgeom.geomID = deviceState()->dcos.geometries.alloc(vgeom);
}

Geometry::~Geometry()
{
  deviceState()->dcos.geometries.free(vgeom.geomID);
}

Geometry *Geometry::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "bezierCurve")
    return new BezierCurve(s);
  if (subtype == "cone")
    return new Cone(s);
//  else if (subtype == "curve")
//    return new Curve(s);
  else if (subtype == "cylinder")
    return new Cylinder(s);
  else if (subtype == "isosurface")
    return new ISOSurface(s);
  else if (subtype == "quad")
    return new Quad(s);
  else if (subtype == "sphere")
    return new Sphere(s);
  else if (subtype == "triangle")
    return new Triangle(s);
  else
    return (Geometry *)new UnknownObject(ANARI_GEOMETRY, s);
}

dco::Geometry Geometry::visionarayGeometry() const
{
  return vgeom;
}

void Geometry::commitParameters()
{
  m_attributes[0] = getParamObject<Array1D>("primitive.attribute0");
  m_attributes[1] = getParamObject<Array1D>("primitive.attribute1");
  m_attributes[2] = getParamObject<Array1D>("primitive.attribute2");
  m_attributes[3] = getParamObject<Array1D>("primitive.attribute3");
  m_attributes[4] = getParamObject<Array1D>("primitive.color");
}

void Geometry::finalize()
{
  for (int i = 0; i < 5; ++i) {
    if (m_attributes[i]) {
      size_t sizeInBytes
          = m_attributes[i]->size() * anari::sizeOf(m_attributes[i]->elementType());

      vattributes[i].resize(sizeInBytes);
      vattributes[i].reset(m_attributes[i]->begin());

      vgeom.primitiveAttributes[i].data = vattributes[i].devicePtr();
      vgeom.primitiveAttributes[i].len = m_attributes[i]->size();
      vgeom.primitiveAttributes[i].typeInfo = getInfo(m_attributes[i]->elementType());
    }
  }
}

void Geometry::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLSCommitSceneRequest =
      helium::newTimeStamp();
}

void Geometry::dispatch()
{
  deviceState()->dcos.geometries.update(vgeom.geomID, vgeom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

// float4 Geometry::getAttributeValue(const Attribute &attr, const Ray &ray) const
// {
//   if (attr == Attribute::NONE)
//     return DEFAULT_ATTRIBUTE_VALUE;
// 
//   auto attrIdx = static_cast<int>(attr);
//   return readAttributeValue(m_attributes[attrIdx].ptr, ray.primID);
// }

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Geometry *);
