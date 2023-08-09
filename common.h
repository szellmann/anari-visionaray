#pragma once

// std
#include <iostream>
// anari
#include <anari/anari_cpp.hpp>
// visionaray
#include "visionaray/math/math.h"

namespace visionaray {
    using uint2 = vec2ui;
    using uint3 = vec3ui;
    using uint4 = vec4ui;
    using float2 = vec2f;
    using float3 = vec3f;
    using float4 = vec4f;
    using Ray = basic_ray<float>;
} // namespace visionaray

namespace anari {

ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint2, ANARI_UINT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint3, ANARI_UINT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint4, ANARI_UINT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float2, ANARI_FLOAT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float3, ANARI_FLOAT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float4, ANARI_FLOAT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::aabb, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::mat4, ANARI_FLOAT32_MAT4);

#ifdef HELIDE_ANARI_DEFINITIONS
ANARI_TYPEFOR_DEFINITION(visionaray::uint2);
ANARI_TYPEFOR_DEFINITION(visionaray::uint3);
ANARI_TYPEFOR_DEFINITION(visionaray::uint4);
ANARI_TYPEFOR_DEFINITION(visionaray::float2);
ANARI_TYPEFOR_DEFINITION(visionaray::float3);
ANARI_TYPEFOR_DEFINITION(visionaray::float4);
ANARI_TYPEFOR_DEFINITION(visionaray::aabb);
ANARI_TYPEFOR_DEFINITION(visionaray::mat4);
#endif

} // namespace anari
