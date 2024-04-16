#pragma once

#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
#include <visionaray/thin_lens_camera.h>

struct TiltShiftCamera : visionaray::thin_lens_camera
{
  TiltShiftCamera() = default;
  TiltShiftCamera(anari::Device device);
  ~TiltShiftCamera();

  anari::Camera getAnariHandle() const
  { return anariCamera; }

  void viewAll(std::array<anari::math::float3, 2> bounds);

  void commit();
 private:
  anari::Device anariDevice{nullptr};
  anari::Camera anariCamera{nullptr};

  // [-1,1]
  float horizontalShift{0.f};

  // [-1,1]
  float verticalShift{0.f};

  // [-PI/2,PI/2]
  float horizontalTilt{0.f};

  // [-PI/2,PI/2]
  float verticalTilt{0.f};
};
