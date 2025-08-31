// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "Object.h"
#include "array/Array1D.h"
#include "array/Array2D.h"
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
                   const DeviceObjectRegistry &DD,
                   unsigned worldID, int frameID)
  {
    if (type == Raycast) {
      asRaycast.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    } else if (type == DirectLight) {
      asDirectLight.renderFrame(
          frame, cam, size, state, DD, rendererState, worldID, frameID);
    }
  }

  VSNRAY_FUNC
  constexpr bool stochasticRendering() const {
    if (type == Raycast) {
      return asRaycast.stochasticRendering;
    } else if (type == DirectLight) {
      return asDirectLight.stochasticRendering;
    }
    return type != Raycast;
  }

  VSNRAY_FUNC
  bool taa() const {
    return type == DirectLight && rendererState.taaEnabled;
  }

  VSNRAY_FUNC
  int sampleLimit() const {
    return rendererState.sampleLimit;
  }

  union {
    VisionarayRendererRaycast     asRaycast;
    VisionarayRendererDirectLight asDirectLight;
  };

  RendererState rendererState;
};

struct Renderer : public Object
{
  Renderer(VisionarayGlobalState *s);
  virtual ~Renderer() = default;

  virtual void commitParameters() override;
  virtual void finalize() override;

  static Renderer *createInstance(
      std::string_view subtype, VisionarayGlobalState *d);

  VisionarayRenderer &visionarayRenderer() { return vrend; }
  const VisionarayRenderer &visionarayRenderer() const { return vrend; }

  bool stochasticRendering() const { return vrend.stochasticRendering(); }

 protected:
  helium::ChangeObserverPtr<Array1D> m_clipPlanes;
  HostDeviceArray<float4> m_clipPlanesOnDevice;
#ifdef WITH_CUDA
  cuda_texture<vector<4, unorm<8>>, 2> m_bgTexture;
#elif defined(WITH_HIP)
  hip_texture<vector<4, unorm<8>>, 2> m_bgTexture;
#else
  texture<vector<4, unorm<8>>, 2> m_bgTexture;
#endif
  VisionarayRenderer vrend;

  helium::IntrusivePtr<Array2D> m_bgImage;
  float4 m_bgColor{float3{0.f}, 1.f};
  float3 m_ambientColor{1.f, 1.f, 1.f};
  float m_ambientRadiance{0.2f};
  std::string m_renderMode{"default"};
  bool m_gradientShading{false};
  bool m_heatMapEnabled{false};
  float m_heatMapScale{0.1f};
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Renderer *, ANARI_RENDERER);
