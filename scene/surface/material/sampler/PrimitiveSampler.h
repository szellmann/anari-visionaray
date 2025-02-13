
#pragma once

#include "DeviceArray.h"
#include "Sampler.h"
#include "array/Array1D.h"

namespace visionaray {

struct PrimitiveSampler : public Sampler
{
  PrimitiveSampler(VisionarayGlobalState *d);

  bool isValid() const override;
  void commitParameters() override;
  void finalize() override;

 private:
  helium::IntrusivePtr<Array1D> m_array;
  uint32_t m_offset{0};

  HostDeviceArray<uint8_t> varray;
};

} // namespace visionaray
