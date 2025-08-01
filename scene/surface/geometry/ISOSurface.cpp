
#include "ISOSurface.h"

namespace visionaray {

ISOSurface::ISOSurface(VisionarayGlobalState *d)
  : Geometry(d)
  , m_field(this)
  , m_isoValue(this)
{
  vgeom.type = dco::Geometry::ISOSurface;
}

ISOSurface::~ISOSurface()
{
}

void ISOSurface::commitParameters()
{
  Geometry::commitParameters();
  m_field = getParamObject<SpatialField>("field");
  m_isoValue = getParamObject<Array1D>("isovalue");
}

void ISOSurface::finalize()
{
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to implicitISOSurface geometry");
    return;
  }

  if (!m_isoValue || m_isoValue->size() == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no ISO values provided to implicitISOSurface geometry");
    return;
  }

  if (!m_field->isValid()) {
    m_field->finalize();
    m_field->markFinalized();
  }

  if (!m_field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "spatial field not valid on implicitISOSurface geometry");
    return;
  }

  m_isoSurface.resize(1);

  m_isoSurface[0].field = m_field->visionaraySpatialField();
  m_isoSurface[0].bounds = m_field->bounds();
  m_isoSurface[0].numValues = m_isoValue->size();
  m_isoSurface[0].values = m_isoValue->beginAs<float>();

  vgeom.primitives.data = m_isoSurface.devicePtr();
  vgeom.primitives.len = m_isoSurface.size();

  deviceState()->objectUpdates.lastBLSReconstructSceneRequest = helium::newTimeStamp();

  dispatch();
}

bool ISOSurface::isValid() const
{
  return m_field && m_field->isValid() && m_isoValue;
}

} // namespace visionaray
