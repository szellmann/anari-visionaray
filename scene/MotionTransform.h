
#pragma once

#include "DeviceArray.h"
#include "Instance.h"
#include "array/Array1D.h"

namespace visionaray {

struct MotionTransform : public Instance
{
  MotionTransform(VisionarayGlobalState *s);

  void commit() override;

  void visionarayGeometryUpdate() override;

 private:
  helium::ChangeObserverPtr<Array1D> m_motionTransform;
  box1 m_time{0.f, 1.f};

  HostDeviceArray<mat4> m_xfms;
  HostDeviceArray<mat3> m_normalXfms;
  HostDeviceArray<mat3> m_affineInv;
  HostDeviceArray<vec3> m_transInv;
};

} // namespace visionaray
