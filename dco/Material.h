// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "dco/common.h"

namespace visionaray::dco {

// Params used by materials //

struct MaterialParamRGB
{
  float3 rgb;
  unsigned samplerID;
  Attribute attribute;
};

VSNRAY_FUNC
inline MaterialParamRGB createMaterialParamRGB()
{
  MaterialParamRGB param;
  param.rgb = float3(0,0,0);
  param.samplerID = UINT_MAX;
  param.attribute = Attribute::None;
  return param;
}

struct MaterialParamUV
{
  float2 uv;
  unsigned samplerID;
  Attribute attribute;
};

VSNRAY_FUNC
inline MaterialParamUV createMaterialParamUV()
{
  MaterialParamUV param;
  param.uv = float2(0,0);
  param.samplerID = UINT_MAX;
  param.attribute = Attribute::None;
  return param;
}

struct MaterialParamF
{
  float f;
  unsigned samplerID;
  Attribute attribute;
};

VSNRAY_FUNC
inline MaterialParamF createMaterialParamF()
{
  MaterialParamF param;
  param.f = 0.f;
  param.samplerID = UINT_MAX;
  param.attribute = Attribute::None;
  return param;
}

enum class AlphaMode
{
  Opaque, Blend, Mask,
};

// Material //

struct Material
{
  enum Type { Matte, PhysicallyBased, Unknown, };
  Type type;
  unsigned matID;
  union {
    struct {
      MaterialParamRGB color;
      MaterialParamF opacity;
      AlphaMode alphaMode;
      float alphaCutoff;
    } asMatte;
    struct {
      MaterialParamRGB baseColor;
      MaterialParamF opacity;
      MaterialParamF metallic;
      MaterialParamF roughness;
      MaterialParamF anisotropyStrength;
      MaterialParamUV anisotropyDirection;
      MaterialParamF anisotropyRotation;
      MaterialParamF transmission;
      struct {
        unsigned samplerID;
      } normal;
      MaterialParamRGB emissive;
      AlphaMode alphaMode;
      float alphaCutoff;
      MaterialParamF clearcoat;
      MaterialParamF clearcoatRoughness;
      float ior;
      MaterialParamRGB sheenColor;
      MaterialParamF sheenRoughness;
    } asPhysicallyBased;
  };

  VSNRAY_FUNC
  inline bool isEmissive() const {
    return type == PhysicallyBased && (
      rgb_to_luminance(asPhysicallyBased.emissive.rgb) > FLT_MIN ||
      validHandle(asPhysicallyBased.emissive.samplerID));
  }
};

VSNRAY_FUNC
inline Material createMaterial()
{
  Material mat;
  memset(&mat,0,sizeof(mat));
  mat.type = Material::Unknown;
  mat.matID = UINT_MAX;
  return mat;
};

VSNRAY_FUNC
inline Material makeDefaultMaterial()
{
  Material mat;
  mat.type = Material::Matte;
  mat.asMatte.color.rgb = vec3(0,1,0);
  mat.asMatte.color.samplerID = UINT_MAX;
  mat.asMatte.color.attribute = Attribute::None;
  mat.asMatte.opacity.f = 1.f;
  mat.asMatte.opacity.samplerID = UINT_MAX;
  mat.asMatte.opacity.attribute = Attribute::None;
  mat.asMatte.alphaMode = AlphaMode::Opaque;
  mat.asMatte.alphaCutoff = 0.5f;
  return mat;
}

} // namespace visionaray::dco
