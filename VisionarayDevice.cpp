// Copyright 2023-2026 Stefan Zellmann
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

#if defined(WITH_CUDA)
#include "anari_library_visionaray_cuda_queries.h"
#elif defined(WITH_HIP)
#include "anari_library_visionaray_hip_queries.h"
#else
#include "anari_library_visionaray_queries.h"
#endif

namespace visionaray {

// Data Arrays ////////////////////////////////////////////////////////////////

void *VisionarayDevice::mapArray(ANARIArray a)
{
#ifdef WITH_CUDA
  // TODO: set device
#elif defined(WITH_HIP)
  // TODO: set device
#else
  deviceState()->renderingSemaphore.arrayMapAcquire();
#endif
  return helium::BaseDevice::mapArray(a);
}

void VisionarayDevice::unmapArray(ANARIArray a)
{
  helium::BaseDevice::unmapArray(a);
#ifdef WITH_CUDA
  // TODO: set device
#elif defined(WITH_HIP)
  // TODO: set device
#else
  deviceState()->renderingSemaphore.arrayMapRelease();
#endif
}

// API Objects ////////////////////////////////////////////////////////////////

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

ANARICamera VisionarayDevice::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIFrame VisionarayDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry VisionarayDevice::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARIGroup VisionarayDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance VisionarayDevice::newInstance(const char *subtype)
{
  initDevice();
  return (ANARIInstance)Instance::createInstance(subtype, deviceState());
}

ANARILight VisionarayDevice::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARIMaterial VisionarayDevice::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARIRenderer VisionarayDevice::newRenderer(const char *subtype)
{
  initDevice();
  return (ANARIRenderer)Renderer::createInstance(subtype, deviceState());
}

ANARISampler VisionarayDevice::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
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

  // TODO: clear context?!
}

void VisionarayDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing visionaray device (%p)", this);
  auto &state = *deviceState();

  state.anariDevice = (anari::Device)this;

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

int VisionarayDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size, uint32_t mask)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "visionaray" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

VisionarayGlobalState *VisionarayDevice::deviceState() const
{
  return (VisionarayGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace visionaray
