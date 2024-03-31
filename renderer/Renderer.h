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

  void renderFrame(const dco::Frame &frame,
                   const dco::Camera &cam,
                   uint2 size,
                   VisionarayGlobalState *state,
                   const VisionarayGlobalState::DeviceObjectRegistry &DD,
                   unsigned worldID, int frameID)
  {
    if (type == Raycast) {
      asRaycast.renderer.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    } else if (type == DirectLight) {
      asDirectLight.renderer.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    }
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
    return type == DirectLight && rendererState.taaEnabled;
  }

  struct {
    VisionarayRendererRaycast renderer;
  } asRaycast;

  struct {
    VisionarayRendererDirectLight renderer;
  } asDirectLight;

  RendererState rendererState;
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
