// Copyright 2023 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "VisionarayDevice.h"
#include "anari/backend/LibraryImpl.h"

#ifdef WITH_CUDA
#include "anari_library_visionaray_cuda_export.h"
#elif defined(WITH_HIP)
#include "anari_library_visionaray_hip_export.h"
#else
#include "anari_library_visionaray_export.h"
#endif

namespace visionaray {

const char **query_extensions();

struct VisionarayLibrary : public anari::LibraryImpl
{
  VisionarayLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

VisionarayLibrary::VisionarayLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice VisionarayLibrary::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new VisionarayDevice(this_library());
}

const char **VisionarayLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace visionaray

// Define library entrypoint //////////////////////////////////////////////////

#ifdef WITH_CUDA
extern "C" VISIONARAY_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    visionaray_cuda, handle, scb, scbPtr)
{
  return (ANARILibrary) new visionaray::VisionarayLibrary(handle, scb, scbPtr);
}
#elif defined(WITH_HIP)
extern "C" VISIONARAY_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    visionaray_hip, handle, scb, scbPtr)
{
  return (ANARILibrary) new visionaray::VisionarayLibrary(handle, scb, scbPtr);
}
#else
extern "C" VISIONARAY_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    visionaray, handle, scb, scbPtr)
{
  return (ANARILibrary) new visionaray::VisionarayLibrary(handle, scb, scbPtr);
}
#endif

extern "C" VISIONARAY_DEVICE_INTERFACE ANARIDevice anariNewVisionarayDevice(
    ANARIStatusCallback defaultCallback, const void *userPtr)
{
  return (ANARIDevice) new visionaray::VisionarayDevice(defaultCallback, userPtr);
}
