// Copyright 2023-2026 Stefan Zellmann
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

// Extra indirection forces VISRTX_LIBRARY_NAME to expand before the entrypoint
// macro pastes it into the symbol name (## suppresses expansion of its operands).
#define VSNRAY_ENTRYPOINT_EXPAND(name, h, s, sp)                               \
  ANARI_DEFINE_LIBRARY_ENTRYPOINT(name, h, s, sp)

#ifdef WITH_CUDA

#ifndef ANARI_VISIONARAY_CUDA_LIBRARY_NAME
#define ANARI_VISIONARAY_CUDA_LIBRARY_NAME visionaray_cuda
#endif

#define VSNRAY_LIBRARY_ENTRYPOINT(h, s, sp)                                    \
  VSNRAY_ENTRYPOINT_EXPAND(ANARI_VISIONARAY_CUDA_LIBRARY_NAME, h, s, sp)

#elif defined(WITH_HIP)

#ifndef ANARI_VISIONARAY_HIP_LIBRARY_NAME
#define ANARI_VISIONARAY_HIP_LIBRARY_NAME visionaray_hip
#endif

#define VSNRAY_LIBRARY_ENTRYPOINT(h, s, sp)                                    \
  VSNRAY_ENTRYPOINT_EXPAND(ANARI_VISIONARAY_LIBRARY_NAME, h, s, sp)

extern "C" VISIONARAY_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    ANARI_VISIONARAY_HIP_LIBRARY_NAME, handle, scb, scbPtr)
{
  return (ANARILibrary) new visionaray::VisionarayLibrary(handle, scb, scbPtr);
}
#else
#ifndef ANARI_VISIONARAY_LIBRARY_NAME
#define ANARI_VISIONARAY_LIBRARY_NAME visionaray
#endif

#define VSNRAY_LIBRARY_ENTRYPOINT(h, s, sp)                                    \
  VSNRAY_ENTRYPOINT_EXPAND(ANARI_VISIONARAY_LIBRARY_NAME, h, s, sp)

#endif

extern "C" VISIONARAY_DEVICE_INTERFACE VSNRAY_LIBRARY_ENTRYPOINT(handle, scb, scbPtr)
{
  return (ANARILibrary) new visionaray::VisionarayLibrary(handle, scb, scbPtr);
}

extern "C" VISIONARAY_DEVICE_INTERFACE ANARIDevice anariNewVisionarayDevice(
    ANARIStatusCallback defaultCallback, const void *userPtr)
{
  return (ANARIDevice) new visionaray::VisionarayDevice(defaultCallback, userPtr);
}
