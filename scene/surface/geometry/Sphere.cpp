// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Sphere.h"

namespace visionaray {

Sphere::Sphere(VisionarayGlobalState *s)
  : Geometry(s)
  , m_BVH(s)
  , m_index(this)
  , m_vertexPosition(this)
  , m_vertexRadius(this)
{
  vgeom.type = dco::Geometry::Sphere;
}

void Sphere::commitParameters()
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
  m_globalRadius = getParam<float>("radius", 0.01f);
}

void Sphere::finalize()
{
  Geometry::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  const float *radius = nullptr;
  if (m_vertexRadius)
    radius = m_vertexRadius->beginAs<float>();

  const auto numSpheres = m_index ? m_index->size() : m_vertexPosition->size();

  m_spheres.resize(numSpheres);

  if (m_index) {
    const auto *indices = m_index->beginAs<uint32_t>();
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<numSpheres; ++i) {
      const auto &v = vertices[indices[i]];
      const float r = radius ? radius[i] : m_globalRadius;
      m_spheres[i].prim_id = i;
      m_spheres[i].geom_id = -1;
      m_spheres[i].center = v;
      m_spheres[i].radius = r;
    }
  } else {
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<numSpheres; ++i) {
      const auto &v = vertices[i];
      const float r = radius ? radius[i] : m_globalRadius;
      m_spheres[i].prim_id = i;
      m_spheres[i].geom_id = -1;
      m_spheres[i].center = v;
      m_spheres[i].radius = r;
    }
  }

  vgeom.primitives.data = m_spheres.devicePtr();
  vgeom.primitives.len = m_spheres.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<uint32_t>());

    vgeom.index.data = vindex.devicePtr();
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

  m_BVH.update((const dco::Sphere *)vgeom.primitives.data,
               vgeom.primitives.len,
               BVH_FLAG_ENABLE_SPATIAL_SPLITS);

  vBLS.type = dco::BLS::Sphere;
#if defined(WITH_CUDA) || defined(WITH_HIP)
  vBLS.asSphere = m_BVH.deviceIndexBVH2();
#else
  vBLS.asSphere = m_BVH.deviceBVH4();
#endif

  dispatch();
}

} // namespace visionaray
