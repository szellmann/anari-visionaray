
#pragma once

// ours
#include "Geometry.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"

namespace visionaray {

struct ISOSurface : public Geometry
{
  ISOSurface(VisionarayGlobalState *d);
  ~ISOSurface() override;

  void commit() override;

  bool isValid() const override;

 private:

  // Data //

  helium::CommitObserverPtr<SpatialField> m_field;
  helium::CommitObserverPtr<Array1D> m_isoValue;
};

} // namespace visionaray
