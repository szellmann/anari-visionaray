#pragma once

// visionaray
#include <visionaray/aligned_vector.h>
#include <visionaray/texture/texture.h>
//ours
#include "array/Array2D.h"
#include "Light.h"

namespace visionaray {

struct HDRI : public Light
{
  HDRI(VisionarayGlobalState *s);
  ~HDRI() override;

  void commit() override;

  // -1 indicates that neither HDRI is
  // used to retrieve background colors
  static int backgroundID;
 private:
  float3 m_up{0.f, 0.f, 1.f};
  float3 m_direction{1.f, 0.f, 0.f};
  helium::IntrusivePtr<Array2D> m_radiance;
  float m_scale{1.f};

  texture<float3, 2> m_radianceTexture;
  aligned_vector<float> m_cdfRows;
  aligned_vector<float> m_cdfLastCol;
};

} // namespace visionaray
