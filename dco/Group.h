// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "dco/common.h"
#include "dco/BLS.h"
#include "dco/Light.h"

namespace visionaray::dco {

// Group //

struct Group
{
  unsigned groupID;

  unsigned numBLSs;
  BLS *BLSs;
  unsigned numGeoms;
  Handle *geoms;
  unsigned numMaterials;
  Handle *materials;
  unsigned numVolumes;
  Handle *volumes;
  unsigned numLights;
  Handle *lights;
  uint32_t *objIds; // surface IDs, volume IDs, etc.
  unsigned numObjIds;
};

VSNRAY_FUNC
inline Group createGroup()
{
  Group group;
  memset(&group,0,sizeof(group));
  group.groupID = UINT_MAX;
  return group;
}

// World //

struct World
{
  unsigned worldID;

  // flat list of lights with instances associated
  LightRef *allLights;

  LightSampler lightSampler;

  VSNRAY_FUNC
  inline unsigned numLights() const
  { return lightSampler.numLights(); };
};

VSNRAY_FUNC
inline World createWorld()
{
  World world;
  world.worldID = UINT_MAX;
  world.allLights = nullptr;
  world.lightSampler = createLightSampler();
  return world;
}

} // namespace visionaray::dco
