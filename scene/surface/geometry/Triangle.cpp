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
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
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

  vgeom.asTriangle.data = m_triangles.devicePtr();
  vgeom.asTriangle.len = m_triangles.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint3>());

    vgeom.asTriangle.index.data = vindex.devicePtr();
    vgeom.asTriangle.index.len = m_index->size();
    vgeom.asTriangle.index.typeInfo = getInfo(m_index->elementType());
  }

  if (m_vertexNormal) {
    vnormals.resize(m_vertexNormal->size());
    vnormals.reset(m_vertexNormal->beginAs<float3>());

    vgeom.asTriangle.normal.data = vnormals.devicePtr();
    vgeom.asTriangle.normal.len = m_vertexNormal->size();
    vgeom.asTriangle.normal.typeInfo = getInfo(m_vertexNormal->elementType());
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_vertexAttributes[i]) {
      size_t sizeInBytes
          = m_vertexAttributes[i]->size()
          * anari::sizeOf(m_vertexAttributes[i]->elementType());

      vattributes[i].resize(sizeInBytes);
      vattributes[i].reset(m_vertexAttributes[i]->begin());

      vgeom.asTriangle.vertexAttributes[i].data = vattributes[i].devicePtr();
      vgeom.asTriangle.vertexAttributes[i].len = m_vertexAttributes[i]->size();
      vgeom.asTriangle.vertexAttributes[i].typeInfo
          = getInfo(m_vertexAttributes[i]->elementType());
    }
  }

  dispatch();
}

void Triangle::cleanup()
{
  m_triangles.clear();
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace visionaray
