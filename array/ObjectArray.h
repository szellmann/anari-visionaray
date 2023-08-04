// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "array/Array1D.h"

namespace visionaray {

struct ObjectArray : public Array
{
  ObjectArray(VisionarayGlobalState *state, const Array1DMemoryDescriptor &d);
  ~ObjectArray();

  void commit() override;

  size_t totalSize() const override;
  size_t totalCapacity() const override;

  size_t size() const;

  void privatize() override;
  void unmap() override;

  Object **handlesBegin() const;
  Object **handlesEnd() const;

  void appendHandle(Object *);
  void removeAppendedHandles();

 private:
  void updateInternalHandleArrays() const;

  mutable std::vector<Object *> m_appendedHandles;
  mutable std::vector<Object *> m_appHandles;
  mutable std::vector<Object *> m_liveHandles;
  size_t m_capacity{0};
  size_t m_begin{0};
  size_t m_end{0};
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::ObjectArray *, ANARI_ARRAY1D);
