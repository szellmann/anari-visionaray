// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "PrimitiveSampler.h"

namespace visionaray {

PrimitiveSampler::PrimitiveSampler(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Primitive;
}

bool PrimitiveSampler::isValid() const
{
  return Sampler::isValid() && m_array;
}

void PrimitiveSampler::commitParameters()
{
  Sampler::commitParameters();
  m_array = getParamObject<Array1D>("array");
  m_offset =
      uint32_t(getParam<uint64_t>("offset", getParam<uint32_t>("offset", 0)));
}

void PrimitiveSampler::finalize()
{
  varray.resize(anari::sizeOf(m_array->elementType()) * m_array->size());
  varray.reset(m_array->data());

  vsampler.asPrimitive.typeInfo = getInfo(m_array->elementType());
  vsampler.asPrimitive.len = m_array->size(); // num elements!
  vsampler.asPrimitive.data = varray.devicePtr();
  vsampler.asPrimitive.offset = m_offset;

  Sampler::dispatch();
}

} // namespace visionaray
