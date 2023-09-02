#pragma once

namespace visionaray {

struct PRD
{
  int x, y;
  uint2 frameSize;
};

struct PixelSample
{
  float4 color;
  float depth;
};

struct RendererState
{
  float4 bgColor{float3(0.f), 1.f};
  float ambientRadiance{1.f};
};

} // visionaray
