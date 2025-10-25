// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <string>
// ours
#include "DeviceCopyableObjects.h"

namespace visionaray {

inline
dco::Attribute toAttribute(std::string str)
{
  using dco::Attribute;
  Attribute res = Attribute::None;

  if (str == "attribute0")
    res = Attribute::_0;
  else if (str == "attribute1")
    res = Attribute::_1;
  else if (str == "attribute2")
    res = Attribute::_2;
  else if (str == "attribute3")
    res = Attribute::_3;
  else if (str == "color")
    res = Attribute::Color;
  else if (str == "worldPosition")
    res = Attribute::WorldPos;
  else if (str == "worldNormal")
    res = Attribute::WorldNormal;
  else if (str == "objectPosition")
    res = Attribute::ObjectPos;
  else if (str == "objectNormal")
    res = Attribute::ObjectNormal;
  else if (str == "none")
    res = Attribute::None;

  return res;
}

inline
dco::AlphaMode toAlphaMode(std::string str)
{
  using dco::AlphaMode;
  AlphaMode res = AlphaMode::Opaque;

  if (str == "opaque")
    res = AlphaMode::Opaque;
  else if (str == "blend")
    res = AlphaMode::Blend;
  else if (str == "mask")
    res = AlphaMode::Mask;

  return res;
}

inline
tex_address_mode toAddressMode(std::string str)
{
  tex_address_mode res;

  if (str == "clampToEdge")
    res = Clamp;
  else if (str == "repeat")
    return Wrap;
  else if (str == "mirrorRepeat")
    return Mirror;

  return res;
}

} // namespace visionaray
