// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Triangle.h"
// std
#include <cstring>

namespace visionaray {

Triangle::Triangle(VisionarayGlobalState *s) : Geometry(s)
{
  vgeom.type = dco::Geometry::Triangle;
}

void Triangle::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  m_vertexPosition->addCommitObserver(this);
  if (m_index)
    m_index->addCommitObserver(this);

  if (m_index) {
    m_triangles.resize(m_index->size());
    for (size_t i=0; i<m_index->size(); ++i) {
      const uint3 idx = m_index->beginAs<uint3>()[i];
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      m_triangles[i].prim_id = i;
      m_triangles[i].geom_id = -1;
      m_triangles[i].v1 = v1;
      m_triangles[i].e1 = v2-v1;
      m_triangles[i].e2 = v3-v1;
    }
  } else {
    m_triangles.resize(m_vertexPosition->size() / 3);
    for (size_t i=0; i<m_triangles.size(); ++i) {
      const uint3 idx(i*3, i*3+1, i*3+2);
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      m_triangles[i].prim_id = i;
      m_triangles[i].geom_id = -1;
      m_triangles[i].v1 = v1;
      m_triangles[i].e1 = v2-v1;
      m_triangles[i].e2 = v3-v1;
    }
  }

  vgeom.asTriangle.data = m_triangles.data();
  vgeom.asTriangle.len = m_triangles.size();

  if (m_index) {
    vgeom.asTriangle.index.data = m_index->begin();
    vgeom.asTriangle.index.len = m_index->size();
    vgeom.asTriangle.index.type = m_index->elementType();
  }

  if (m_vertexAttributes[4]) {
    vgeom.asTriangle.vertex.color.data = m_vertexAttributes[4]->begin();
    vgeom.asTriangle.vertex.color.len = m_vertexAttributes[4]->size();
    vgeom.asTriangle.vertex.color.type = m_vertexAttributes[4]->elementType();
  }
}

// float4 Triangle::getAttributeValue(const Attribute &attr, const Ray &ray) const
// {
//   if (attr == Attribute::NONE)
//     return DEFAULT_ATTRIBUTE_VALUE;
// 
//   auto attrIdx = static_cast<int>(attr);
//   auto *attributeArray = m_vertexAttributes[attrIdx].ptr;
//   if (!attributeArray)
//     return Geometry::getAttributeValue(attr, ray);
// 
//   const float3 uvw(1.0f - ray.u - ray.v, ray.u, ray.v);
// 
//   auto idx = m_index ? *(m_index->dataAs<uint3>() + ray.primID)
//                      : 3 * ray.primID + uint3(0, 1, 2);
// 
//   auto a = readAttributeValue(attributeArray, idx.x);
//   auto b = readAttributeValue(attributeArray, idx.y);
//   auto c = readAttributeValue(attributeArray, idx.z);
// 
//   return uvw.x * a + uvw.y * b + uvw.z * c;
// }

void Triangle::cleanup()
{
  m_triangles.clear();
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace visionaray
