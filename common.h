#pragma once

// std
#include <iostream>
// anari
#include <anari/anari_cpp.hpp>
// visionaray
#include "visionaray/math/math.h"

// ==================================================================
// common traits
// ==================================================================

namespace visionaray {

struct TypeInfo
{
  ANARIDataType dataType{ANARI_UNKNOWN};
  unsigned sizeInBytes{0};
  unsigned numComponents{0};
  bool fixed{false};
  bool sRGB{false};
};

VSNRAY_FUNC
constexpr TypeInfo getInfo(ANARIDataType type)
{
  TypeInfo ti;

  if (type == ANARI_UFIXED8) {
    ti.dataType = type;
    ti.sizeInBytes = 1;
    ti.numComponents = 1;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED8_VEC2) {
    ti.dataType = type;
    ti.sizeInBytes = 2;
    ti.numComponents = 2;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED8_VEC3) {
    ti.dataType = type;
    ti.sizeInBytes = 3;
    ti.numComponents = 3;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED8_VEC4) {
    ti.dataType = type;
    ti.sizeInBytes = 4;
    ti.numComponents = 4;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED16) {
    ti.dataType = type;
    ti.sizeInBytes = 2;
    ti.numComponents = 1;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED16_VEC2) {
    ti.dataType = type;
    ti.sizeInBytes = 4;
    ti.numComponents = 2;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED16_VEC3) {
    ti.dataType = type;
    ti.sizeInBytes = 6;
    ti.numComponents = 3;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED16_VEC4) {
    ti.dataType = type;
    ti.sizeInBytes = 8;
    ti.numComponents = 4;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED32) {
    ti.dataType = type;
    ti.sizeInBytes = 4;
    ti.numComponents = 1;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED32_VEC2) {
    ti.dataType = type;
    ti.sizeInBytes = 8;
    ti.numComponents = 2;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED32_VEC3) {
    ti.dataType = type;
    ti.sizeInBytes = 12;
    ti.numComponents = 3;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_UFIXED32_VEC4) {
    ti.dataType = type;
    ti.sizeInBytes = 16;
    ti.numComponents = 4;
    ti.fixed = true;
    ti.sRGB = false;
  }
  else if (type == ANARI_FLOAT32) {
    ti.dataType = type;
    ti.sizeInBytes = 4;
    ti.numComponents = 1;
    ti.fixed = false;
    ti.sRGB = false;
  }
  else if (type == ANARI_FLOAT32_VEC2) {
    ti.dataType = type;
    ti.sizeInBytes = 8;
    ti.numComponents = 2;
    ti.fixed = false;
    ti.sRGB = false;
  }
  else if (type == ANARI_FLOAT32_VEC3) {
    ti.dataType = type;
    ti.sizeInBytes = 12;
    ti.numComponents = 3;
    ti.fixed = false;
    ti.sRGB = false;
  }
  else if (type == ANARI_FLOAT32_VEC4) {
    ti.dataType = type;
    ti.sizeInBytes = 16;
    ti.numComponents = 4;
    ti.fixed = false;
    ti.sRGB = false;
  }
  return ti;
}

VSNRAY_FUNC
inline void convert(vec4 *dest, const uint8_t *source, const TypeInfo &ti)
{
  if (ti.fixed) {
    switch (ti.dataType) {
      case ANARI_UFIXED8: {
        unorm<8> u8;
        memcpy(&u8, source, sizeof(u8));
        (*dest).x = float(u8);
        (*dest).y = 0.f;
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED8_VEC2: {
        vector<2, unorm<8>> u8;
        memcpy(&u8, source, sizeof(u8));
        (*dest).x = float(u8.x);
        (*dest).y = float(u8.y);
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED8_VEC3: {
        vector<3, unorm<8>> u8;
        memcpy(&u8, source, sizeof(u8));
        (*dest).x = float(u8.x);
        (*dest).y = float(u8.y);
        (*dest).z = float(u8.z);
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED8_VEC4: {
        vector<4, unorm<8>> u8;
        memcpy(&u8, source, sizeof(u8));
        (*dest).x = float(u8.x);
        (*dest).y = float(u8.y);
        (*dest).z = float(u8.z);
        (*dest).w = float(u8.w);
        break;
      }
      case ANARI_UFIXED16: {
        unorm<16> u16;
        memcpy(&u16, source, sizeof(u16));
        (*dest).x = float(u16);
        (*dest).y = 0.f;
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED16_VEC2: {
        vector<2, unorm<16>> u16;
        memcpy(&u16, source, sizeof(u16));
        (*dest).x = float(u16.x);
        (*dest).y = float(u16.y);
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED16_VEC3: {
        vector<3, unorm<16>> u16;
        memcpy(&u16, source, sizeof(u16));
        (*dest).x = float(u16.x);
        (*dest).y = float(u16.y);
        (*dest).z = float(u16.z);
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED16_VEC4: {
        vector<4, unorm<16>> u16;
        memcpy(&u16, source, sizeof(u16));
        (*dest).x = float(u16.x);
        (*dest).y = float(u16.y);
        (*dest).z = float(u16.z);
        (*dest).w = float(u16.w);
        break;
      }
      case ANARI_UFIXED32: {
        unorm<32> u32;
        memcpy(&u32, source, sizeof(u32));
        (*dest).x = float(u32);
        (*dest).y = 0.f;
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED32_VEC2: {
        vector<2, unorm<32>> u32;
        memcpy(&u32, source, sizeof(u32));
        (*dest).x = float(u32.x);
        (*dest).y = float(u32.y);
        (*dest).z = 0.f;
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED32_VEC3: {
        vector<3, unorm<32>> u32;
        memcpy(&u32, source, sizeof(u32));
        (*dest).x = float(u32.x);
        (*dest).y = float(u32.y);
        (*dest).z = float(u32.z);
        (*dest).w = 1.f;
        break;
      }
      case ANARI_UFIXED32_VEC4: {
        vector<4, unorm<32>> u32;
        memcpy(&u32, source, sizeof(u32));
        (*dest).x = float(u32.x);
        (*dest).y = float(u32.y);
        (*dest).z = float(u32.z);
        (*dest).w = float(u32.w);
        break;
      }
    }
  } else {
    *dest = vec4(0.f, 0.f, 0.f, 1.f);
    memcpy(dest, source, ti.sizeInBytes);
  }

  if (ti.sRGB) {
    for (unsigned c=0; c<ti.numComponents; ++c) {

    }
  }
}

} // namespace visionaray


// ==================================================================
// math types
// ==================================================================

namespace visionaray {
  using int2 = vec2i;
  using int3 = vec3i;
  using int4 = vec4i;
  using uint2 = vec2ui;
  using uint3 = vec3ui;
  using uint4 = vec4ui;
  using float2 = vec2f;
  using float3 = vec3f;
  using float4 = vec4f;
} // namespace visionaray

namespace anari {

ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint2, ANARI_UINT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint3, ANARI_UINT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::uint4, ANARI_UINT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float2, ANARI_FLOAT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float3, ANARI_FLOAT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::float4, ANARI_FLOAT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::box1, ANARI_FLOAT32_BOX1);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::box2, ANARI_FLOAT32_BOX2);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::aabb, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::aabbi, ANARI_INT32_BOX3);
ANARI_TYPEFOR_SPECIALIZATION(visionaray::mat4, ANARI_FLOAT32_MAT4);

#ifdef HELIDE_ANARI_DEFINITIONS
ANARI_TYPEFOR_DEFINITION(visionaray::uint2);
ANARI_TYPEFOR_DEFINITION(visionaray::uint3);
ANARI_TYPEFOR_DEFINITION(visionaray::uint4);
ANARI_TYPEFOR_DEFINITION(visionaray::float2);
ANARI_TYPEFOR_DEFINITION(visionaray::float3);
ANARI_TYPEFOR_DEFINITION(visionaray::float4);
ANARI_TYPEFOR_DEFINITION(visionaray::box1);
ANARI_TYPEFOR_DEFINITION(visionaray::box2);
ANARI_TYPEFOR_DEFINITION(visionaray::aabb);
ANARI_TYPEFOR_DEFINITION(visionaray::aabbi);
ANARI_TYPEFOR_DEFINITION(visionaray::mat4);
#endif

} // namespace anari
