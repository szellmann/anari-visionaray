// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "DeviceArray.h"
#include "DeviceCopyableObjects.h"

namespace visionaray
{
struct DeviceCopyableObjects
{
  // One TLS per world
  DeviceObjectArray<dco::TLS> TLSs;
  DeviceObjectArray<dco::World> worlds; // TODO: move TLSs and EPS in here!
  DeviceObjectArray<dco::Group> groups;
  DeviceObjectArray<dco::Surface> surfaces;
  DeviceObjectArray<dco::Instance> instances;
  DeviceObjectArray<dco::Geometry> geometries;
  DeviceObjectArray<dco::Material> materials;
  DeviceObjectArray<dco::Sampler> samplers;
  DeviceObjectArray<dco::Volume> volumes;
  DeviceObjectArray<dco::SpatialField> spatialFields;
  DeviceObjectArray<dco::Light> lights;
  DeviceObjectArray<dco::Frame> frames;
};

struct DeviceObjectRegistry
{
  dco::TLS *TLSs{nullptr};
  dco::World *worlds{nullptr};
  dco::Group *groups{nullptr};
  dco::Surface *surfaces{nullptr};
  dco::Instance *instances{nullptr};
  dco::Geometry *geometries{nullptr};
  dco::Material *materials{nullptr};
  dco::Sampler *samplers{nullptr};
  dco::Volume *volumes{nullptr};
  dco::SpatialField *spatialFields{nullptr};
  dco::Light *lights{nullptr};
  dco::Frame *frames{nullptr};
};

} // visionaray
