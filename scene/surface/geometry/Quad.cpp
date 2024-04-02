// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Quad.h"

namespace visionaray {

Quad::Quad(VisionarayGlobalState *s)
  : Geometry(s)
  , m_index(this)
  , m_vertexPosition(this)
  , m_vertexNormal(this)
  , m_vertexTangent(this)
{
  vgeom.type = dco::Geometry::Quad;
}

void Quad::commit()
{
  Geometry::commit();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_vertexTangent = getParamObject<Array1D>("vertex.tangent");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on quad geometry");
    return;
  }

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
      m_triangles[i*2].e2 = v4-v1;
      m_triangles[i*2+1].prim_id = i;
      m_triangles[i*2+1].geom_id = -1;
      m_triangles[i*2+1].v1 = v3;
      m_triangles[i*2+1].e1 = v4-v3;
      m_triangles[i*2+1].e2 = v2-v3;
    }
  } else {
    size_t numQuads = m_vertexPosition->size() / 4;
    m_triangles.resize(numQuads * 2);
    for (size_t i=0; i<numQuads; ++i) {
      const uint4 idx(i*4, i*4+1, i*4+2, i*4+3);
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      const vec3f v4 = m_vertexPosition->beginAs<vec3f>()[idx.w];
      m_triangles[i*2].prim_id = i;
      m_triangles[i*2].geom_id = -1;
      m_triangles[i*2].v1 = v1;
      m_triangles[i*2].e1 = v2-v1;
      m_triangles[i*2].e2 = v4-v1;
      m_triangles[i*2+1].prim_id = i;
      m_triangles[i*2+1].geom_id = -1;
      m_triangles[i*2+1].v1 = v3;
      m_triangles[i*2+1].e1 = v4-v3;
      m_triangles[i*2+1].e2 = v2-v3;
    }
  }

  vgeom.asQuad.data = m_triangles.devicePtr();
  vgeom.asQuad.len = m_triangles.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint3>());

    vgeom.asQuad.index.data = vindex.devicePtr();
    vgeom.asQuad.index.len = m_index->size();
    vgeom.asQuad.index.typeInfo = getInfo(m_index->elementType());
  }

  if (m_vertexNormal) {
    vnormals.resize(m_vertexNormal->size());
    vnormals.reset(m_vertexNormal->beginAs<float3>());

    vgeom.asQuad.normal.data = vnormals.devicePtr();
    vgeom.asQuad.normal.len = m_vertexNormal->size();
    vgeom.asQuad.normal.typeInfo = getInfo(m_vertexNormal->elementType());
  }

  if (m_vertexTangent) {
    vtangents.resize(m_vertexTangent->size());
    if (m_vertexTangent->elementType() == ANARI_FLOAT32_VEC4) {
      vtangents.reset(m_vertexTangent->beginAs<float4>());
    } else if (m_vertexTangent->elementType() == ANARI_FLOAT32_VEC3) {
      for (size_t i = 0; i < m_vertexTangent->size(); ++i) {
        float3 tng = m_vertexTangent->beginAs<float3>()[i];
        vtangents[i] = float4(tng, 1.f);
      }
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "unsupported type for 'vertex.tangent' on quad geometry");
    }

    vgeom.asQuad.tangent.data = vtangents.devicePtr();
    vgeom.asQuad.tangent.len = m_vertexTangent->size();
    vgeom.asQuad.tangent.typeInfo = getInfo(m_vertexTangent->elementType());
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_vertexAttributes[i]) {
      size_t sizeInBytes
          = m_vertexAttributes[i]->size()
          * anari::sizeOf(m_vertexAttributes[i]->elementType());

      vattributes[i].resize(sizeInBytes);
      vattributes[i].reset(m_vertexAttributes[i]->begin());

      vgeom.asQuad.vertexAttributes[i].data = vattributes[i].devicePtr();
      vgeom.asQuad.vertexAttributes[i].len = m_vertexAttributes[i]->size();
      vgeom.asQuad.vertexAttributes[i].typeInfo
          = getInfo(m_vertexAttributes[i]->elementType());
    }
  }

  dispatch();
}

} // namespace visionaray
