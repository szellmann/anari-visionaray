
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

  return res;
}

} // namespace visionaray
