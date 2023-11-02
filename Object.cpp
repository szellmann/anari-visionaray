// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Object.h"
// std
#include <atomic>
#include <cstdarg>

namespace visionaray {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(ANARIDataType type, VisionarayGlobalState *s)
    : helium::BaseObject(type, s)
{
  helium::BaseObject::markUpdated();
}

void Object::commit()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

bool Object::isValid() const
{
  return true;
}

VisionarayGlobalState *Object::deviceState() const
{
  return (VisionarayGlobalState *)helium::BaseObject::m_state;
}

// UnknownObject definitions //////////////////////////////////////////////////

UnknownObject::UnknownObject(ANARIDataType type, VisionarayGlobalState *s)
    : Object(type, s)
{
  s->objectCounts.unknown++;
}

UnknownObject::~UnknownObject()
{
  deviceState()->objectCounts.unknown--;
}

bool UnknownObject::isValid() const
{
  return false;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Object *);
