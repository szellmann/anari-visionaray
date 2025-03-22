// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "World.h"

namespace visionaray {

World::World(VisionarayGlobalState *s)
    : Object(ANARI_WORLD, s),
      m_zeroSurfaceData(this),
      m_zeroVolumeData(this),
      m_zeroLightData(this),
      m_instanceData(this)
{
  m_zeroGroup = new Group(s);
  m_zeroInstance = new Instance(s);
  m_zeroInstance->setParamDirect("group", m_zeroGroup.ptr);

  // never any public ref to these objects
  m_zeroGroup->refDec(helium::RefType::PUBLIC);
  m_zeroInstance->refDec(helium::RefType::PUBLIC);
}

World::~World()
{
  cleanup();
}

bool World::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      visionaraySceneUpdate();
    }
    if (vscene && vscene->isValid()) {
      auto bounds = vscene->getBounds();
      std::memcpy(ptr, &bounds, sizeof(bounds));
      return true;
    }
  }

  return Object::getProperty(name, type, ptr, flags);
}

void World::commitParameters()
{
  m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<ObjectArray>("volume");
  m_zeroLightData = getParamObject<ObjectArray>("light");
}

void World::finalize()
{
  cleanup();

  const bool addZeroInstance =
      m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData;
  if (addZeroInstance)
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visionaray::World will add zero instance");

  if (m_zeroSurfaceData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visionaray::World found %zu surfaces in zero instance",
        m_zeroSurfaceData->size());
    m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
  } else
    m_zeroGroup->removeParam("surface");

  if (m_zeroVolumeData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visionaray::World found %zu volumes in zero instance",
        m_zeroVolumeData->size());
    m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
  } else
    m_zeroGroup->removeParam("volume");

  if (m_zeroLightData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visionaray::World found %zu lights in zero instance",
        m_zeroLightData->size());
    m_zeroGroup->setParamDirect("light", getParamDirect("light"));
  } else
    m_zeroGroup->removeParam("light");

  m_zeroGroup->commitParameters();
  m_zeroInstance->commitParameters();
  m_zeroGroup->finalize();
  m_zeroInstance->finalize();

  m_instanceData = getParamObject<ObjectArray>("instance");

  m_instances.clear();

  if (m_instanceData) {
    m_instanceData->removeAppendedHandles();
    if (addZeroInstance)
      m_instanceData->appendHandle(m_zeroInstance.ptr);
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid())
            m_instances.push_back((Instance *)o);
        });
  } else if (addZeroInstance)
    m_instances.push_back(m_zeroInstance.ptr);

  m_objectUpdates.lastTLSBuild = 0;
  m_objectUpdates.lastBLSReconstructCheck = 0;
  m_objectUpdates.lastBLSCommitCheck = 0;
}

const std::vector<Instance *> &World::instances() const
{
  return m_instances;
}

VisionarayScene World::visionarayScene() const
{
  return vscene;
}

void World::visionaraySceneUpdate()
{
  rebuildBLSs();
  recommitBLSs();
  rebuildTLS();
}

void World::rebuildBLSs()
{
  const auto &state = *deviceState();
  if (state.objectUpdates.lastBLSReconstructSceneRequest
      < m_objectUpdates.lastBLSReconstructCheck) {
    return;
  }

  m_objectUpdates.lastTLSBuild = 0; // BLS changed, so need to build TLS
  reportMessage(ANARI_SEVERITY_DEBUG,
      "visionaray::World rebuilding %zu BLSs",
      m_instances.size());
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    inst->group()->visionaraySceneConstruct();
  });

  m_objectUpdates.lastBLSReconstructCheck = helium::newTimeStamp();
  m_objectUpdates.lastBLSCommitCheck = helium::newTimeStamp();
}

void World::recommitBLSs()
{
  const auto &state = *deviceState();
  if (state.objectUpdates.lastBLSCommitSceneRequest
      < m_objectUpdates.lastBLSCommitCheck) {
    return;
  }

  m_objectUpdates.lastTLSBuild = 0; // BLS changed, so need to build TLS
  reportMessage(ANARI_SEVERITY_DEBUG,
      "visionaray::World recommitting %zu BLSs",
      m_instances.size());
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    inst->group()->visionaraySceneCommit();
  });

  m_objectUpdates.lastBLSCommitCheck = helium::newTimeStamp();
}

void World::rebuildTLS()
{
  const auto &state = *deviceState();
  if (state.objectUpdates.lastTLSReconstructSceneRequest
      < m_objectUpdates.lastTLSBuild) {
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "visionaray::World rebuilding TLS over %zu instances",
      m_instances.size());

  if (vscene)
    vscene->release();
  vscene = newVisionarayScene(VisionaraySceneImpl::World, deviceState());

  uint32_t id = 0;
  uint32_t lightID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *i) {
    if (i && i->isValid()
        && (!i->group()->surfaces().empty() || !i->group()->volumes().empty()
            || !i->group()->lights().empty())) {
      i->visionarayInstanceUpdate();
      vscene->attachInstance(i->visionarayInstance(), id, i->id());
    } else if (i->group()->surfaces().empty() && i->group()->volumes().empty()
        && !i->group()->lights().empty()) {
      // Only lights - these will not end up on a valid instance
      // (as there's no BLS), therefore add them here
      // TODO: we should transform those lights..
      std::for_each(i->group()->lights().begin(),
          i->group()->lights().end(),
          [&](auto *o) {
            auto *l = (Light *)o;
            if (l && l->isValid()) {
              vscene->attachLight(l->visionarayLight(), lightID++);
            } else {
              reportMessage(
                  ANARI_SEVERITY_DEBUG, "    visionaray::Light is invalid");
            }
          });
    } else {
      if (i->group()->surfaces().empty()) {
        reportMessage(ANARI_SEVERITY_DEBUG,
            "visionaray::World rejecting empty surfaces in instance(%p) "
            "when building TLS",
            i);
      } else {
        reportMessage(ANARI_SEVERITY_DEBUG,
            "visionaray::World rejecting invalid surfaces in instance(%p) "
            "when building TLS",
            i);
      }
    }
    id++;
  });

  vscene->commit();
  m_objectUpdates.lastTLSBuild = helium::newTimeStamp();
}

void World::cleanup()
{
  if (vscene)
    vscene->release();
  vscene = nullptr;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::World *);
