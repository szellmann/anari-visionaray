// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "VisionarayDevice.h"

#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "array/ObjectArray.h"
#include "frame/Frame.h"
#include "scene/light/Light.h"
#include "scene/surface/geometry/Geometry.h"
#include "scene/surface/material/sampler/Sampler.h"
#include "scene/surface/Surface.h"
#include "scene/volume/spatial_field/SpatialField.h"
#include "scene/volume/Volume.h"
#include "scene/Group.h"
#include "scene/Instance.h"

namespace visionaray {

///////////////////////////////////////////////////////////////////////////////
// Generated function declarations ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const char **query_object_types(ANARIDataType type);

const void *query_object_info(ANARIDataType type,
    const char *subtype,
    const char *infoName,
    ANARIDataType infoType);

const void *query_param_info(ANARIDataType type,
    const char *subtype,
    const char *paramName,
    ANARIDataType paramType,
    const char *infoName,
    ANARIDataType infoType);

const char **query_extensions();

///////////////////////////////////////////////////////////////////////////////
// VisionarayDevice definitions ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Data Arrays ////////////////////////////////////////////////////////////////

ANARIArray1D VisionarayDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

  Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new Array1D(deviceState(), md);
}

ANARIArray2D VisionarayDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D VisionarayDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

  Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new Array3D(deviceState(), md);
}

// Renderable Objects /////////////////////////////////////////////////////////

ANARILight VisionarayDevice::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARICamera VisionarayDevice::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIGeometry VisionarayDevice::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARISpatialField VisionarayDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)SpatialField::createInstance(subtype, deviceState());
}

ANARISurface VisionarayDevice::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume VisionarayDevice::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

// Surface Meta-Data //////////////////////////////////////////////////////////

ANARIMaterial VisionarayDevice::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARISampler VisionarayDevice::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup VisionarayDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance VisionarayDevice::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld VisionarayDevice::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **VisionarayDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return visionaray::query_object_types(objectType);
}

const void *VisionarayDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return visionaray::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *VisionarayDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return visionaray::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Object + Parameter Lifetime Management /////////////////////////////////////

int VisionarayDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  if (handleIsDevice(object)) {
    std::string_view prop = name;
    if (prop == "extension" && type == ANARI_STRING_LIST) {
      helium::writeToVoidP(mem, query_extensions());
      return 1;
    } else if (prop == "visionaray" && type == ANARI_BOOL) {
      helium::writeToVoidP(mem, true);
      return 1;
    }
  } else {
    if (mask == ANARI_WAIT) {
      deviceState()->waitOnCurrentFrame();
      flushCommitBuffer();
    }
    return helium::referenceFromHandle(object).getProperty(
        name, type, mem, mask);
  }

  return 0;
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame VisionarayDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer VisionarayDevice::newRenderer(const char *subtype)
{
  initDevice();
  return (ANARIRenderer)Renderer::createInstance(subtype, deviceState());
}

// Other VisionarayDevice definitions /////////////////////////////////////////

VisionarayDevice::VisionarayDevice(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<VisionarayGlobalState>(this_device());
  deviceCommitParameters();
}

VisionarayDevice::VisionarayDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<VisionarayGlobalState>(this_device());
  deviceCommitParameters();
}

VisionarayDevice::~VisionarayDevice()
{
  auto &state = *deviceState();

  state.commitBuffer.clear();

  reportMessage(ANARI_SEVERITY_DEBUG, "destroying visionaray device (%p)", this);

  // rtcReleaseDevice(state.embreeDevice);

  // NOTE: These object leak warnings are not required to be done by
  //       implementations as the debug layer in the SDK is far more
  //       comprehensive and designed for detecting bugs like this. However
  //       these simple checks are very straightforward to implement and do not
  //       really add substantial code complexity, so they are provided out of
  //       convenience.

  auto reportLeaks = [&](size_t &count, const char *handleType) {
    if (count != 0) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "detected %zu leaked %s objects",
          count,
          handleType);
    }
  };

  reportLeaks(state.objectCounts.frames, "ANARIFrame");
  reportLeaks(state.objectCounts.cameras, "ANARICamera");
  reportLeaks(state.objectCounts.renderers, "ANARIRenderer");
  reportLeaks(state.objectCounts.worlds, "ANARIWorld");
  reportLeaks(state.objectCounts.instances, "ANARIInstance");
  reportLeaks(state.objectCounts.groups, "ANARIGroup");
  reportLeaks(state.objectCounts.surfaces, "ANARISurface");
  reportLeaks(state.objectCounts.geometries, "ANARIGeometry");
  reportLeaks(state.objectCounts.materials, "ANARIMaterial");
  reportLeaks(state.objectCounts.samplers, "ANARISampler");
  reportLeaks(state.objectCounts.volumes, "ANARIVolume");
  reportLeaks(state.objectCounts.spatialFields, "ANARISpatialField");
  reportLeaks(state.objectCounts.arrays, "ANARIArray");

  if (state.objectCounts.unknown != 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "detected %zu leaked ANARIObject objects created by unknown subtypes",
        state.objectCounts.unknown);
  }
}

void VisionarayDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing visionaray device (%p)", this);

  auto &state = *deviceState();

  // state.embreeDevice = rtcNewDevice(nullptr);

  // if (!state.embreeDevice) {
  //   reportMessage(ANARI_SEVERITY_ERROR,
  //       "Embree error %d - cannot create device\n",
  //       rtcGetDeviceError(nullptr));
  // }

  // rtcSetDeviceErrorFunction(
  //     state.embreeDevice,
  //     [](void *userPtr, RTCError error, const char *str) {
  //       auto *d = (VisionarayDevice *)userPtr;
  //       d->reportMessage(
  //           ANARI_SEVERITY_ERROR, "Embree error %d - '%s'", error, str);
  //     },
  //     this);

  m_initialized = true;
}

void VisionarayDevice::deviceCommitParameters()
{
  auto &state = *deviceState();

  // bool allowInvalidSurfaceMaterials = state.allowInvalidSurfaceMaterials;

  // state.allowInvalidSurfaceMaterials =
  //     getParam<bool>("allowInvalidMaterials", true);
  // state.invalidMaterialColor =
  //     getParam<float4>("invalidMaterialColor", float4(1.f, 0.f, 1.f, 1.f));

  // if (allowInvalidSurfaceMaterials != state.allowInvalidSurfaceMaterials)
  //   state.objectUpdates.lastBLSReconstructSceneRequest = helium::newTimeStamp();

  helium::BaseDevice::deviceCommitParameters();
}

VisionarayGlobalState *VisionarayDevice::deviceState() const
{
  return (VisionarayGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace visionaray
