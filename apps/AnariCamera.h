#pragma once

#include <memory>
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
#include <visionaray/thin_lens_camera.h>

struct AnariCamera : visionaray::thin_lens_camera
{
  typedef std::shared_ptr<AnariCamera> SP;
  AnariCamera() = default;
  AnariCamera(anari::Device device);
  ~AnariCamera();

  anari::Camera getAnariHandle() const
  { return anariCamera; }

  void viewAll(std::array<anari::math::float3, 2> bounds);

  void commit();
 private:
  anari::Device anariDevice{nullptr};
  anari::Camera anariCamera{nullptr};
};
