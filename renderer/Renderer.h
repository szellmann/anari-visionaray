#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
#include "scene/VisionarayScene.h"
// impls
#include "Raycast_impl.h"

namespace visionaray {

struct VisionarayRenderer
{
  enum Type { Raycast, };
  Type type;

  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice) {
    if (type == Raycast) {
      return asRaycast.renderer.renderSample(ray, prd, worldID, onDevice);
    }

    return {};
  }

  struct {
    VisionarayRendererRaycast renderer;
  } asRaycast;
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer visionarayRenderer() const { return vrend; }

 protected:
  VisionarayRenderer vrend;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
