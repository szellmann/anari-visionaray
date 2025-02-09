
#pragma once

#include "DeviceArray.h"
#include "Geometry.h"

namespace visionaray {

struct Cone : public Geometry
{
  Cone(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:

  HostDeviceArray<dco::Cone> m_cones;
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexRadius;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;

  HostDeviceArray<uint2> vindex;
  HostDeviceArray<uint8_t> vattributes[5];
};

} // namespace visionaray
