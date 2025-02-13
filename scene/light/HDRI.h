#pragma once

// visionaray
#include <visionaray/texture/texture.h>
//ours
#include "array/Array2D.h"
#include "DeviceArray.h"
#include "Light.h"

namespace visionaray {

struct HDRI : public Light
{
  HDRI(VisionarayGlobalState *s);
  ~HDRI() override;

  void commitParameters() override;
  void finalize() override;

  // -1 indicates that neither HDRI is
  // used to retrieve background colors
  static int backgroundID;
 private:
  float3 m_up{0.f, 0.f, 1.f};
  float3 m_direction{1.f, 0.f, 0.f};
  helium::IntrusivePtr<Array2D> m_radiance;
  float m_scale{1.f};

#ifdef WITH_CUDA
  cuda_texture<float4, 2> m_radianceTexture;
#elif defined(WITH_HIP)
  hip_texture<float4, 2> m_radianceTexture;
#else
  texture<float4, 2> m_radianceTexture;
#endif
  HostDeviceArray<float> m_cdfRows;
  HostDeviceArray<float> m_cdfLastCol;
};

} // namespace visionaray
