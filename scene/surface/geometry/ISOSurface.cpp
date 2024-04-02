
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

void ISOSurface::commit()
{
  Geometry::commit();

  m_field = getParamObject<SpatialField>("field");
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to implicitISOSurface volume");
    return;
  }

  m_isoValue = getParamObject<Array1D>("isovalue");

  if (!m_isoValue || m_isoValue->size() == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no ISO values provided to implicitISOSurface volume");
    return;
  }

  vgeom.asISOSurface.data.field = m_field->visionaraySpatialField();
  vgeom.asISOSurface.data.bounds = m_field->bounds();
  vgeom.asISOSurface.data.numValues = m_isoValue->size();
  vgeom.asISOSurface.data.values = m_isoValue->beginAs<float>();
  vgeom.updated = true;

  dispatch();
}

bool ISOSurface::isValid() const
{
  return m_field && m_field->isValid() && m_isoValue;
}

} // namespace visionaray
