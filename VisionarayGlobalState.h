#pragma once

// std
#include <atomic>
// helium
#include "helium/BaseGlobalDeviceState.h"
// visionaray
#include "visionaray/detail/thread_pool.h"
// ours
#include "DeviceCopyableObjects.h"
#include "DeviceArray.h"
#include "RenderingSemaphore.h"

namespace visionaray {

struct Frame;

struct VisionarayGlobalState : public helium::BaseGlobalDeviceState
{
  thread_pool threadPool;

  struct ObjectCounts
  {
    std::atomic<size_t> frames{0};
    std::atomic<size_t> cameras{0};
    std::atomic<size_t> renderers{0};
    std::atomic<size_t> worlds{0};
    std::atomic<size_t> instances{0};
    std::atomic<size_t> groups{0};
    std::atomic<size_t> surfaces{0};
    std::atomic<size_t> geometries{0};
    std::atomic<size_t> materials{0};
    std::atomic<size_t> samplers{0};
    std::atomic<size_t> volumes{0};
    std::atomic<size_t> spatialFields{0};
    std::atomic<size_t> lights{0};
    std::atomic<size_t> arrays{0};
    std::atomic<size_t> unknown{0};
  } objectCounts;

  struct ObjectUpdates
  {
    helium::TimeStamp lastBLSReconstructSceneRequest{0};
    helium::TimeStamp lastBLSCommitSceneRequest{0};
    helium::TimeStamp lastTLSReconstructSceneRequest{0};
  } objectUpdates;

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
  } dcos;

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
  } onDevice;

#ifdef WITH_CUDA
  cudaStream_t stream;
#elif defined(WITH_HIP)
  hipStream_t stream;
#else
  RenderingSemaphore renderingSemaphore;
#endif
  Frame *currentFrame{nullptr};

  // Helper methods //

  VisionarayGlobalState(ANARIDevice d);
  ~VisionarayGlobalState();
  void waitOnCurrentFrame() const;
};

// Helper functions/macros ////////////////////////////////////////////////////

inline VisionarayGlobalState *asVisionarayState(helium::BaseGlobalDeviceState *s)
{
  return (VisionarayGlobalState *)s;
}

#define VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISIONARAY_ANARI_TYPEFOR_DEFINITION(type)                              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // visionaray
