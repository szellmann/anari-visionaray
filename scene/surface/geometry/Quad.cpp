// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Quad.h"

namespace visionaray {

Quad::Quad(VisionarayGlobalState *s) : Geometry(s)
{
  vgeom.type = dco::Geometry::Quad;
}

void Quad::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  //m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  //m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  //m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  //m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  //m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on quad geometry");
    return;
  }

  m_vertexPosition->addCommitObserver(this);
  if (m_index)
    m_index->addCommitObserver(this);

  if (m_index) {
    m_triangles.resize(m_index->size() * 2);
    for (size_t i=0; i<m_index->size(); ++i) {
      const uint4 idx = m_index->beginAs<uint4>()[i];
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      const vec3f v4 = m_vertexPosition->beginAs<vec3f>()[idx.w];
      m_triangles[i*2].prim_id = i;
      m_triangles[i*2].geom_id = -1;
      m_triangles[i*2].v1 = v1;
      m_triangles[i*2].e1 = v2-v1;
      m_triangles[i*2].e2 = v3-v1;
      m_triangles[i*2+1].prim_id = i;
      m_triangles[i*2+1].geom_id = -1;
      m_triangles[i*2+1].v1 = v1;
      m_triangles[i*2+1].e1 = v3-v1;
      m_triangles[i*2+1].e2 = v4-v1;
    }
  } else {
    m_triangles.resize(m_vertexPosition->size() / 2);
    for (size_t i=0; i<m_triangles.size(); ++i) {
      const uint4 idx(i*4, i*4+1, i*4+2, i*4+3);
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      const vec3f v4 = m_vertexPosition->beginAs<vec3f>()[idx.w];
      m_triangles[i*2].prim_id = i;
      m_triangles[i*2].geom_id = -1;
      m_triangles[i*2].v1 = v1;
      m_triangles[i*2].e1 = v2-v1;
      m_triangles[i*2].e2 = v3-v1;
      m_triangles[i*2+1].prim_id = i;
      m_triangles[i*2+1].geom_id = -1;
      m_triangles[i*2+1].v1 = v1;
      m_triangles[i*2+1].e1 = v3-v1;
      m_triangles[i*2+1].e2 = v4-v1;
    }
  }

  vgeom.asQuad.data = m_triangles.data();
  vgeom.asQuad.len = m_triangles.size();
}

// float4 Quad::getAttributeValue(const Attribute &attr, const Ray &ray) const
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
//   auto idx = m_index ? *(m_index->dataAs<uint4>() + ray.primID)
//                      : 4 * ray.primID + uint4(0, 1, 2, 3);
// 
//   float4 uv((1 - ray.v) * (1 - ray.u),
//       (1 - ray.v) * ray.u,
//       ray.v * ray.u,
//       ray.v * (1 - ray.u));
// 
//   auto a = readAttributeValue(attributeArray, idx.x);
//   auto b = readAttributeValue(attributeArray, idx.y);
//   auto c = readAttributeValue(attributeArray, idx.z);
//   auto d = readAttributeValue(attributeArray, idx.w);
// 
//   return uv.x * a + uv.y * b + uv.z * c + uv.w * d;
// }

void Quad::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace visionaray
