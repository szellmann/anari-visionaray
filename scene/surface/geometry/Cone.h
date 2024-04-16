
#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct Cone : public Geometry
{
  Cone(VisionarayGlobalState *s);

  void commit() override;

 private:

  HostDeviceArray<dco::Cone> m_cones;
  helium::CommitObserverPtr<Array1D> m_index;
  helium::CommitObserverPtr<Array1D> m_vertexPosition;
  helium::CommitObserverPtr<Array1D> m_vertexRadius;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;

  HostDeviceArray<uint2> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
