// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace visionaray {

// some common functions need by grid builders _and_ traversal code..

VSNRAY_FUNC inline
size_t linearIndex(const vec3i index, const vec3i dims)
{
  return index.z * size_t(dims.x) * dims.y
       + index.y * dims.x
       + index.x;
}

VSNRAY_FUNC inline
vec3i projectOnGrid(const vec3f V,
                    const vec3i dims,
                    const box3f worldBounds)
{
  const vec3f V01 = (V-worldBounds.min)/(worldBounds.max-worldBounds.min);
  return clamp(vec3i(V01*vec3f(dims)),vec3i(0),dims-vec3i(1));
}

} // namespace visionaray
