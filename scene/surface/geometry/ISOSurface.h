
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

  helium::ChangeObserverPtr<SpatialField> m_field;
  helium::ChangeObserverPtr<Array1D> m_isoValue;
};

} // namespace visionaray
