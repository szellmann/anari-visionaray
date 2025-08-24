// Copyright 2023-2025 Stefan Zellmann
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
  helium::BaseObject::markParameterChanged();
  s->commitBuffer.addObjectToCommit(this);
}

void Object::commitParameters()
{
  // no-op
}

void Object::finalize()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
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
{}

bool UnknownObject::isValid() const
{
  return false;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Object *);
