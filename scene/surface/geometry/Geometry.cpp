// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
// subtypes
//#include "Cone.h"
//#include "Curve.h"
#include "Cylinder.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"
// std
#include <cstring>
#include <limits>

namespace visionaray {

Geometry::Geometry(VisionarayGlobalState *s) : Object(ANARI_GEOMETRY, s)
{
  memset(&vgeom,0,sizeof(vgeom));
  s->objectCounts.geometries++;
}

Geometry::~Geometry()
{
//  rtcReleaseGeometry(m_embreeGeometry);
  deviceState()->objectCounts.geometries--;
}

Geometry *Geometry::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
//  if (subtype == "cone")
//    return new Cone(s);
//  else if (subtype == "curve")
//    return new Curve(s);
  /*else*/ if (subtype == "cylinder")
    return new Cylinder(s);
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

void Geometry::commit()
{
  m_attributes[0] = getParamObject<Array1D>("primitive.attribute0");
  m_attributes[1] = getParamObject<Array1D>("primitive.attribute1");
  m_attributes[2] = getParamObject<Array1D>("primitive.attribute2");
  m_attributes[3] = getParamObject<Array1D>("primitive.attribute3");
  m_attributes[4] = getParamObject<Array1D>("primitive.color");

  if (m_attributes[4]) {
    vgeom.primitive.color.data = m_attributes[4]->begin();
    vgeom.primitive.color.len = m_attributes[4]->size();
    vgeom.primitive.color.type = m_attributes[4]->elementType();
  }
}

void Geometry::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLSCommitSceneRequest =
      helium::newTimeStamp();
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
