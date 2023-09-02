// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Raycast.h"
#include "DirectLight.h"
#include "Renderer.h"

namespace visionaray {

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(VisionarayGlobalState *s) : Object(ANARI_RENDERER, s)
{
  s->objectCounts.renderers++;

  Array1DMemoryDescriptor md;
  md.elementType = ANARI_FLOAT32_VEC3;
  md.numItems = 4;
}

Renderer::~Renderer()
{
  deviceState()->objectCounts.renderers--;
}

void Renderer::commit()
{
  auto &raycastState = vrend.asRaycast.renderer.rendererState;

  raycastState.bgColor = getParam<float4>("background", float4(float3(0.f), 1.f));
  raycastState.ambientRadiance = getParam<float>("ambientRadiance", 1.f);
}

Renderer *Renderer::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "raycast")
    return new Raycast(s);
  else if (subtype == "direct_light" || subtype == "default")
    return new DirectLight(s);
  else
    return {};
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Renderer *);
