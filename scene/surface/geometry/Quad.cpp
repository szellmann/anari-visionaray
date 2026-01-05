// Copyright 2023-2026 Stefan Zellmann
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

void Quad::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_vertexTangent = getParamObject<Array1D>("vertex.tangent");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
}

void Quad::finalize()
{
  Geometry::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on quad geometry");
    return;
  }

  unsigned nextID = 0;
  auto addTriangles = [&](basic_triangle<3, float> &tri1,
                          basic_triangle<3, float> &tri2) {
    //if (length(tri1.e1) > 0.f && length(tri1.e2) > 0.f &&
    //    length(tri2.e1) > 0.f && length(tri2.e2) > 0.f) {
    //if (length(tri.e1) > 0.f && length(tri.e2) > 0.f) {
    // TODO: we can discard invalid primitives here, but
    // then also have to make sure to also discard vertex
    // attributes (at least, iff we don't have indices!)
    if (true) {
      unsigned primID = nextID++;
      tri1.prim_id = primID;
      tri2.prim_id = primID;
      m_triangles.push_back(tri1);
      m_triangles.push_back(tri2);
    }
  };

  if (m_index) {
    for (size_t i=0; i<m_index->size(); ++i) {
      const uint4 idx = m_index->beginAs<uint4>()[i];
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      const vec3f v4 = m_vertexPosition->beginAs<vec3f>()[idx.w];
      basic_triangle<3, float> tri1, tri2;
      tri1.geom_id = -1;
      tri1.v1 = v1;
      tri1.e1 = v2-v1;
      tri1.e2 = v4-v1;
      tri2.geom_id = -1;
      tri2.v1 = v3;
      tri2.e1 = v4-v3;
      tri2.e2 = v2-v3;
      addTriangles(tri1, tri2);
    }
  } else {
    size_t numQuads = m_vertexPosition->size() / 4;
    for (size_t i=0; i<numQuads; ++i) {
      const uint4 idx(i*4, i*4+1, i*4+2, i*4+3);
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      const vec3f v4 = m_vertexPosition->beginAs<vec3f>()[idx.w];
      basic_triangle<3, float> tri1, tri2;
      tri1.geom_id = -1;
      tri1.v1 = v1;
      tri1.e1 = v2-v1;
      tri1.e2 = v4-v1;
      tri2.geom_id = -1;
      tri2.v1 = v3;
      tri2.e1 = v4-v3;
      tri2.e2 = v2-v3;
      addTriangles(tri1, tri2);
    }
  }

  vgeom.primitives.data = m_triangles.devicePtr();
  vgeom.primitives.len = m_triangles.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint4>());

    vgeom.index.data = vindex.devicePtr();
    vgeom.index.len = m_index->size();
    vgeom.index.typeInfo = getInfo(m_index->elementType());
  }

  if (m_vertexNormal) {
    vnormals.resize(m_vertexNormal->size());
    vnormals.reset(m_vertexNormal->beginAs<float3>());

    vgeom.normal.data = vnormals.devicePtr();
    vgeom.normal.len = m_vertexNormal->size();
    vgeom.normal.typeInfo = getInfo(m_vertexNormal->elementType());
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

    vgeom.tangent.data = vtangents.devicePtr();
    vgeom.tangent.len = m_vertexTangent->size();
    vgeom.tangent.typeInfo = getInfo(m_vertexTangent->elementType());
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_vertexAttributes[i]) {
      size_t sizeInBytes
          = m_vertexAttributes[i]->size()
          * anari::sizeOf(m_vertexAttributes[i]->elementType());

      vattributes[i].resize(sizeInBytes);
      vattributes[i].reset(m_vertexAttributes[i]->begin());

      vgeom.vertexAttributes[i].data = vattributes[i].devicePtr();
      vgeom.vertexAttributes[i].len = m_vertexAttributes[i]->size();
      vgeom.vertexAttributes[i].typeInfo
          = getInfo(m_vertexAttributes[i]->elementType());
    }
  }

  dispatch();
}

} // namespace visionaray
