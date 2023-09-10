#pragma once

// helium
#include "helium/BaseGlobalDeviceState.h"
// visionaray
#include "visionaray/detail/thread_pool.h"
// ours
#include "DeviceCopyableObjects.h"

namespace visionaray {

struct Frame;

struct VisionarayGlobalState : public helium::BaseGlobalDeviceState
{
  thread_pool threadPool;

  struct ObjectCounts
  {
    size_t frames{0};
    size_t cameras{0};
    size_t renderers{0};
    size_t worlds{0};
    size_t instances{0};
    size_t groups{0};
    size_t surfaces{0};
    size_t geometries{0};
    size_t materials{0};
    size_t samplers{0};
    size_t volumes{0};
    size_t spatialFields{0};
    size_t lights{0};
    size_t arrays{0};
    size_t unknown{0};
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
    std::vector<dco::TLS> TLSs;
    std::vector<dco::Group> groups;
    std::vector<dco::Instance> instances;
    std::vector<dco::Material> materials;
    std::vector<dco::SpatialField> spatialFields;
    std::vector<dco::GridAccel> gridAccels;
    std::vector<dco::TransferFunction> transferFunctions;
    std::vector<dco::Light> lights;
  } dcos;

  struct DeviceObjectRegistry
  {
    dco::TLS *TLSs{nullptr};
    dco::Group *groups{nullptr};
    dco::Instance *instances{nullptr};
    dco::Material *materials{nullptr};
    dco::SpatialField *spatialFields{nullptr};
    dco::GridAccel *gridAccels{nullptr};
    dco::TransferFunction *transferFunctions{nullptr};
    dco::Light *lights{nullptr};
  } onDevice;

  Frame *currentFrame{nullptr};

  // Helper methods //

  VisionarayGlobalState(ANARIDevice d);
  void waitOnCurrentFrame() const;
};

} // visionaray
