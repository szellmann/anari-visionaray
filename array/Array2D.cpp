// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "array/Array2D.h"

namespace visionaray {

Array2D::Array2D(VisionarayGlobalState *state, const Array2DMemoryDescriptor &d)
    : Array(ANARI_ARRAY2D, state, d)
{
  m_size[0] = d.numItems1;
  m_size[1] = d.numItems2;

  initManagedMemory();
}

size_t Array2D::totalSize() const
{
  return size(0) * size(1);
}

size_t Array2D::size(int dim) const
{
  return m_size[dim];
}

uint2 Array2D::size() const
{
  return uint2(uint32_t(size(0)), uint32_t(size(1)));
}

void Array2D::privatize()
{
  makePrivatizedCopy(size(0) * size(1));
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Array2D *);
