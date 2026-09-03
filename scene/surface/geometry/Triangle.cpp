// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Triangle.h"
// std
#include <cstring>

namespace visionaray {

Triangle::Triangle(VisionarayGlobalState *s)
  : Geometry(s)
  , m_BVH(s)
  , m_index(this)
  , m_vertexPosition(this)
  , m_vertexNormal(this)
  , m_vertexTangent(this)
  , m_faceVaryingNormal(this)
  , m_faceVaryingTangent(this)
{
  vgeom.type = dco::Geometry::Triangle;
}

float Triangle::surfaceArea() const
{
  return m_surfaceArea;
}

void Triangle::commitParameters()
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

  m_faceVaryingNormal = getParamObject<Array1D>("faceVarying.normal");
  m_faceVaryingTangent = getParamObject<Array1D>("faceVarying.tangent");
  m_faceVaryingAttributes[0] = getParamObject<Array1D>("faceVarying.attribute0");
  m_faceVaryingAttributes[1] = getParamObject<Array1D>("faceVarying.attribute1");
  m_faceVaryingAttributes[2] = getParamObject<Array1D>("faceVarying.attribute2");
  m_faceVaryingAttributes[3] = getParamObject<Array1D>("faceVarying.attribute3");
  m_faceVaryingAttributes[4] = getParamObject<Array1D>("faceVarying.color");
}

void Triangle::finalize()
{
  Geometry::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  m_triangles.clear();
  m_surfaceArea = 0.f;

  unsigned nextID = 0;
  auto addTriangle = [&](dco::Triangle &tri) {
    //if (length(tri.e1) > 0.f && length(tri.e2) > 0.f) {
    // TODO: we can discard invalid primitives here, but
    // then also have to make sure to also discard vertex
    // attributes (at least, iff we don't have indices!)
    if (true) {
      tri.prim_id = nextID++;
      m_triangles.push_back(tri);
      m_surfaceArea += area(tri);
    }
  };

  if (m_index) {
    for (size_t i=0; i<m_index->size(); ++i) {
      const uint3 idx = m_index->beginAs<uint3>()[i];
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      dco::Triangle triangle;
      triangle.geom_id = -1;
      triangle.v1 = v1;
      triangle.e1 = v2-v1;
      triangle.e2 = v3-v1;
      addTriangle(triangle);
    }
  } else {
    for (size_t i=0; i<m_vertexPosition->size() / 3; ++i) {
      const uint3 idx(i*3, i*3+1, i*3+2);
      const vec3f v1 = m_vertexPosition->beginAs<vec3f>()[idx.x];
      const vec3f v2 = m_vertexPosition->beginAs<vec3f>()[idx.y];
      const vec3f v3 = m_vertexPosition->beginAs<vec3f>()[idx.z];
      dco::Triangle triangle;
      triangle.geom_id = -1;
      triangle.v1 = v1;
      triangle.e1 = v2-v1;
      triangle.e2 = v3-v1;
      addTriangle(triangle);
    }
  }

  vgeom.primitives.data = m_triangles.devicePtr();
  vgeom.primitives.len = m_triangles.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint3>());

    vgeom.index.data = vindex.devicePtr();
    vgeom.index.len = m_index->size();
    vgeom.index.typeInfo = getInfo(m_index->elementType());
  }

  if (m_vertexNormal) {
    vnormals.resize(m_vertexNormal->size());
    vnormals.reset(m_vertexNormal->beginAs<float3>());

    vgeom.vertex.normal.data = vnormals.devicePtr();
    vgeom.vertex.normal.len = m_vertexNormal->size();
    vgeom.vertex.normal.typeInfo = getInfo(m_vertexNormal->elementType());
  }

  // per vertex
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
          "unsupported type for 'vertex.tangent' on triangle geometry");
    }

    vgeom.vertex.tangent.data = vtangents.devicePtr();
    vgeom.vertex.tangent.len = m_vertexTangent->size();
    vgeom.vertex.tangent.typeInfo = getInfo(m_vertexTangent->elementType());
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_vertexAttributes[i]) {
      size_t sizeInBytes
          = m_vertexAttributes[i]->size()
          * anari::sizeOf(m_vertexAttributes[i]->elementType());

      vattributes[i].resize(sizeInBytes);
      vattributes[i].reset(m_vertexAttributes[i]->begin());

      vgeom.vertex.attributes[i].data = vattributes[i].devicePtr();
      vgeom.vertex.attributes[i].len = m_vertexAttributes[i]->size();
      vgeom.vertex.attributes[i].typeInfo
          = getInfo(m_vertexAttributes[i]->elementType());
    }
  }

  // face-varying
  if (m_faceVaryingNormal) {
    fvnormals.resize(m_faceVaryingNormal->size());
    fvnormals.reset(m_faceVaryingNormal->beginAs<float3>());

    vgeom.faceVarying.normal.data = fvnormals.devicePtr();
    vgeom.faceVarying.normal.len = m_faceVaryingNormal->size();
    vgeom.faceVarying.normal.typeInfo = getInfo(m_faceVaryingNormal->elementType());
  }

  if (m_faceVaryingTangent) {
    fvtangents.resize(m_faceVaryingTangent->size());
    if (m_faceVaryingTangent->elementType() == ANARI_FLOAT32_VEC4) {
      fvtangents.reset(m_faceVaryingTangent->beginAs<float4>());
    } else if (m_faceVaryingTangent->elementType() == ANARI_FLOAT32_VEC3) {
      for (size_t i = 0; i < m_faceVaryingTangent->size(); ++i) {
        float3 tng = m_faceVaryingTangent->beginAs<float3>()[i];
        fvtangents[i] = float4(tng, 1.f);
      }
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "unsupported type for 'faceVarying.tangent' on triangle geometry");
    }

    vgeom.faceVarying.tangent.data = fvtangents.devicePtr();
    vgeom.faceVarying.tangent.len = m_faceVaryingTangent->size();
    vgeom.faceVarying.tangent.typeInfo = getInfo(m_faceVaryingTangent->elementType());
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_faceVaryingAttributes[i]) {
      size_t sizeInBytes
          = m_faceVaryingAttributes[i]->size()
          * anari::sizeOf(m_faceVaryingAttributes[i]->elementType());

      fvattributes[i].resize(sizeInBytes);
      fvattributes[i].reset(m_faceVaryingAttributes[i]->begin());

      vgeom.faceVarying.attributes[i].data = fvattributes[i].devicePtr();
      vgeom.faceVarying.attributes[i].len = m_faceVaryingAttributes[i]->size();
      vgeom.faceVarying.attributes[i].typeInfo
          = getInfo(m_faceVaryingAttributes[i]->elementType());
    }
  }

  m_BVH.update((const dco::Triangle *)vgeom.primitives.data,
               vgeom.primitives.len,
               BVH_FLAG_ENABLE_SPATIAL_SPLITS | BVH_FLAG_NO_STREAM_SYNCHRONIZE);

  vBLS.type = dco::BLS::Triangle;
#if defined(WITH_CUDA) || defined(WITH_HIP)
  vBLS.asTriangle = m_BVH.deviceBVH2();
#else
  vBLS.asTriangle = m_BVH.deviceBVH4();
#endif

  dispatch();
}

} // namespace visionaray
