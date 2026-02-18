// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Cone.h"

namespace visionaray {

Cone::Cone(VisionarayGlobalState *s)
  : Geometry(s)
  , m_index(this)
  , m_vertexPosition(this)
  , m_vertexRadius(this)
{
  vgeom.type = dco::Geometry::Cone;
}

void Cone::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
}

void Cone::finalize()
{
  Geometry::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cone geometry");
    return;
  }

  if (!m_vertexRadius) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.radius' on cone geometry");
    return;
  }

  const auto numCones =
      m_index ? m_index->size() : m_vertexPosition->size() / 2;

  m_cones.resize(numCones);

  if (m_index) {
    const auto *indices = m_index->beginAs<uint2>();
    const auto *vertices = m_vertexPosition->beginAs<float3>();
    const auto *radii = m_vertexRadius->beginAs<float>();

    for (size_t i=0; i<numCones; ++i) {
      const auto &v1 = vertices[indices[i].x];
      const auto &v2 = vertices[indices[i].y];
      const float r1 = radii[indices[i].x];
      const float r2 = radii[indices[i].y];
      m_cones[i].prim_id = i;
      m_cones[i].geom_id = -1;
      m_cones[i].v1 = v1;
      m_cones[i].v2 = v2;
      m_cones[i].r1 = r1;
      m_cones[i].r2 = r2;
    }
  } else {
    const auto *vertices = m_vertexPosition->beginAs<float3>();
    const auto *radii = m_vertexRadius->beginAs<float>();

    for (size_t i=0; i<numCones; ++i) {
      const auto &v1 = vertices[i*2];
      const auto &v2 = vertices[i*2+1];
      const float r1 = radii[i*2];
      const float r2 = radii[i*2+1];
      m_cones[i].prim_id = i;
      m_cones[i].geom_id = -1;
      m_cones[i].v1 = v1;
      m_cones[i].v2 = v2;
      m_cones[i].r1 = r1;
      m_cones[i].r2 = r2;
    }
  }

  vgeom.primitives.data = m_cones.devicePtr();
  vgeom.primitives.len = m_cones.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint2>());

    vgeom.index.data = m_index->begin();
    vgeom.index.len = m_index->size();
    vgeom.index.typeInfo = getInfo(m_index->elementType());
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

  m_BVH.update((const dco::Cone *)vgeom.primitives.data,
               vgeom.primitives.len,
               &deviceState()->threadPool,
               0); // no spatial splits for cones yet!

  vBLS.type = dco::BLS::Cone;
#if defined(WITH_CUDA) || defined(WITH_HIP)
  vBLS.asCone = m_BVH.deviceIndexBVH2();
#else
  vBLS.asCone = m_BVH.deviceBVH4();
#endif

  dispatch();
}

} // namespace visionaray
