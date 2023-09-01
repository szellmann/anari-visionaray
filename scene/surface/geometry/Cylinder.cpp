// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Cylinder.h"
// std
#include <numeric>

namespace visionaray {

Cylinder::Cylinder(VisionarayGlobalState *s) : Geometry(s)
{
  vgeom.type = dco::Geometry::Cylinder;
}

void Cylinder::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  //m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  //m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  //m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  //m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  //m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cylinder geometry");
    return;
  }

  if (m_index)
    m_index->addCommitObserver(this);
  m_vertexPosition->addCommitObserver(this);

  const float *radius = m_radius ? m_radius->beginAs<float>() : nullptr;
  m_globalRadius = getParam<float>("radius", 1.f);

  const auto numCylinders =
      m_index ? m_index->size() : m_vertexPosition->size() / 2;

  m_cylinders.resize(numCylinders);  

  if (m_index) {
    const auto *indices = m_index->beginAs<uint2>();
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<numCylinders; ++i) {
      const auto &v1 = vertices[indices[i].x];
      const auto &v2 = vertices[indices[i].y];
      const float r = radius ? radius[i] : m_globalRadius;
      m_cylinders[i].prim_id = i;
      m_cylinders[i].geom_id = -1;
      m_cylinders[i].v1 = v1;
      m_cylinders[i].v2 = v2;
      m_cylinders[i].radius = r;
    }
  } else {
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<numCylinders; ++i) {
      const auto &v1 = vertices[i*2];
      const auto &v2 = vertices[i*2+1];
      const float r = radius ? radius[i] : m_globalRadius;
      m_cylinders[i].prim_id = i;
      m_cylinders[i].geom_id = -1;
      m_cylinders[i].v1 = v1;
      m_cylinders[i].v2 = v2;
      m_cylinders[i].radius = r;
    }
  }

  vgeom.asCylinder.data = m_cylinders.data();
  vgeom.asCylinder.len = m_cylinders.size();
}

// float4 Cylinder::getAttributeValue(const Attribute &attr, const Ray &ray) const
// {
//   if (attr == Attribute::NONE)
//     return DEFAULT_ATTRIBUTE_VALUE;
// 
//   auto attrIdx = static_cast<int>(attr);
//   auto *attributeArray = m_vertexAttributes[attrIdx].ptr;
//   if (!attributeArray)
//     return Geometry::getAttributeValue(attr, ray);
// 
//   auto idx = m_index ? *(m_index->dataAs<uint2>() + ray.primID)
//                      : 2 * ray.primID + uint2(0, 1);
// 
//   auto a = readAttributeValue(attributeArray, idx.x);
//   auto b = readAttributeValue(attributeArray, idx.y);
// 
//   return a + (b - a) * ray.u;
// }

void Cylinder::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace visionaray
