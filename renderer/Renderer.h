#pragma once

#include "Object.h"
#include "array/Array1D.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
// impls
#include "Raycast_impl.h"
#include "DirectLight_impl.h"

namespace visionaray {

struct VisionarayRenderer
{
  enum Type { Raycast, DirectLight, };
  Type type;

  VSNRAY_FUNC
  PixelSample renderSample(ScreenSample &ss, Ray ray, unsigned worldID,
        const VisionarayGlobalState::DeviceObjectRegistry &onDevice) const {
    if (type == Raycast) {
      return asRaycast.renderer.renderSample(ss, ray, worldID, onDevice);
    } else if (type == DirectLight) {
      return asDirectLight.renderer.renderSample(ss, ray, worldID, onDevice);
    }

    return {};
  }

  VSNRAY_FUNC
  constexpr bool stochasticRendering() const {
    if (type == Raycast) {
      return asRaycast.renderer.stochasticRendering;
    } else if (type == DirectLight) {
      return asDirectLight.renderer.stochasticRendering;
    }
    return type != Raycast;
  }

  VSNRAY_FUNC
  bool taa() const {
    return type == DirectLight && rendererState().taaEnabled;
  }

  VSNRAY_FUNC
  int spp() const {
    if (type == DirectLight)
      return asDirectLight.renderer.rendererState.pixelSamples;
    else
      return 1;
  }

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
