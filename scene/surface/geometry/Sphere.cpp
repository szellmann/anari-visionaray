// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Sphere.h"

namespace visionaray {

Sphere::Sphere(VisionarayGlobalState *s) : Geometry(s)
{
  vgeom.type = dco::Geometry::Sphere;
}

void Sphere::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  m_vertexPosition->addCommitObserver(this);
  if (m_vertexRadius)
    m_vertexRadius->addCommitObserver(this);
  if (m_index)
    m_index->addCommitObserver(this);

  m_globalRadius = getParam<float>("radius", 0.01f);

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

  vgeom.asSphere.data = m_spheres.data();
  vgeom.asSphere.len = m_spheres.size();

  if (m_index) {
    vgeom.asSphere.index.data = m_index->begin();
    vgeom.asSphere.index.len = m_index->size();
    vgeom.asSphere.index.type = m_index->elementType();
  }

  for (int i = 0; i < 5; ++i ) {
    if (m_vertexAttributes[i]) {
      vgeom.asSphere.vertexAttributes[i].data = m_vertexAttributes[i]->begin();
      vgeom.asSphere.vertexAttributes[i].len = m_vertexAttributes[i]->size();
      vgeom.asSphere.vertexAttributes[i].type = m_vertexAttributes[i]->elementType();
    }
  }

  dispatch();
}

// float4 Sphere::getAttributeValue(const Attribute &attr, const Ray &ray) const
// {
//   if (attr == Attribute::NONE)
//     return Geometry::getAttributeValue(attr, ray);
// 
//   auto attrIdx = static_cast<int>(attr);
//   auto *attributeArray = m_vertexAttributes[attrIdx].ptr;
//   if (!attributeArray)
//     return Geometry::getAttributeValue(attr, ray);
// 
//   const auto primID =
//       m_attributeIndex.empty() ? ray.primID : m_attributeIndex[ray.primID];
// 
//   return readAttributeValue(attributeArray, primID);
// }

void Sphere::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
  if (m_vertexRadius)
    m_vertexRadius->removeCommitObserver(this);
}

} // namespace visionaray
