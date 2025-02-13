#pragma once

#include "Camera.h"

namespace visionaray {

struct Orthographic : public Camera
{
  Orthographic(VisionarayGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:
   float m_aspect{1.f};
   float m_height{1.f};
   float3 m_dir_du;
   float3 m_dir_dv;
   float3 m_dir_00;
};

} // namespace visionaray
