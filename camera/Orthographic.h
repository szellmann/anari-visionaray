#pragma once

#include "Camera.h"

namespace visionaray {

struct Orthographic : public Camera
{
  Orthographic(VisionarayGlobalState *s);

  void commit() override;

 private:
   float3 m_dir_du;
   float3 m_dir_dv;
   float3 m_dir_00;
};

} // namespace visionaray
