#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
#include "scene/VisionarayScene.h"
// impls
#include "Raycast_impl.h"
#include "DirectLight_impl.h"

namespace visionaray {

struct VisionarayRenderer
{
  enum Type { Raycast, DirectLight, };
  Type type;

  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts objCounts) {
    if (type == Raycast) {
      return asRaycast.renderer.renderSample(ray, prd, worldID, onDevice, objCounts);
    } else if (type == DirectLight) {
      return asDirectLight.renderer.renderSample(ray, prd, worldID, onDevice, objCounts);
    }

    return {};
  }

  VSNRAY_FUNC
  bool stochasticRendering() const {
    return type != Raycast;
  }

  VSNRAY_FUNC
  int spp() const {
//  return type == Raycast ? 1 : 4;
    return 1;
  };

  VSNRAY_FUNC
  const RendererState &rendererState() const {
    if (type == Raycast)
      return asRaycast.renderer.rendererState;
    else
      return asDirectLight.renderer.rendererState;
  }

  VSNRAY_FUNC
  const RendererState &constRendererState() const {
    return rendererState();
  }

  VSNRAY_FUNC
  RendererState &rendererState() {
    return const_cast<RendererState &>(constRendererState());
  }

  struct {
    VisionarayRendererRaycast renderer;
  } asRaycast;

  struct {
    VisionarayRendererDirectLight renderer;
  } asDirectLight;
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  ~Renderer() override;

  virtual void commit() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer &visionarayRenderer() { return vrend; }
  const VisionarayRenderer &visionarayRenderer() const { return vrend; }

  bool stochasticRendering() const { return vrend.stochasticRendering(); }

 protected:
  VisionarayRenderer vrend;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
