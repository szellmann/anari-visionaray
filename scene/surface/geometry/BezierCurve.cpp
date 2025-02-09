#include "BezierCurve.h"

namespace visionaray {

BezierCurve::BezierCurve(VisionarayGlobalState *s)
  : Geometry(s)
  , m_index(this)
  , m_radius(this)
  , m_vertexPosition(this)
{
  vgeom.type = dco::Geometry::BezierCurve;
}

void BezierCurve::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
}

void BezierCurve::finalize()
{
  Geometry::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on bezierCurve geometry");
    return;
  }

  const float *radius = m_radius ? m_radius->beginAs<float>() : nullptr;
  m_globalRadius = getParam<float>("radius", 1.f);

  if (m_index) {
    const auto *indices = m_index->beginAs<unsigned>();
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<m_index->size(); ++i) {
      unsigned first = indices[i];
      unsigned last  = (i == m_index->size()-1) ? first+3 : indices[i+1];

      for (size_t j=first; j<last; j+=3) {
        const auto &w0 = vertices[j];
        const auto &w1 = vertices[j+1];
        const auto &w2 = vertices[j+2];
        const auto &w3 = vertices[j+3];
        const float r = radius ? radius[i] : m_globalRadius;

        dco::BezierCurve curve;
        curve.prim_id = i;
        curve.geom_id = -1;
        curve.w0 = w0;
        curve.w1 = w1;
        curve.w2 = w2;
        curve.w3 = w3;
        curve.r  = r;
        m_curves.push_back(curve);
      }
    }
  } else {
    const auto *vertices = m_vertexPosition->beginAs<float3>();

    for (size_t i=0; i<m_vertexPosition->size()-3; i+=3) {
      const auto &w0 = vertices[i];
      const auto &w1 = vertices[i+1];
      const auto &w2 = vertices[i+2];
      const auto &w3 = vertices[i+3];
      const float r = radius ? radius[i/3] : m_globalRadius;

      dco::BezierCurve curve;
      curve.prim_id = i/3;
      curve.geom_id = -1;
      curve.w0 = w0;
      curve.w1 = w1;
      curve.w2 = w2;
      curve.w3 = w3;
      curve.r  = r;
      m_curves.push_back(curve);
    }
  }

  vgeom.primitives.data = m_curves.devicePtr();
  vgeom.primitives.len = m_curves.size();

  if (m_index) {
    vindex.resize(m_index->size());
    vindex.reset(m_index->beginAs<unsigned>());

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

  dispatch();
}

} // namespace visionaray
