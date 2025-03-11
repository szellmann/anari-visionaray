// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
// std
#include <iterator>

namespace visionaray {

Group::Group(VisionarayGlobalState *s)
  : Object(ANARI_GROUP, s)
  , m_surfaceData(this)
  , m_volumeData(this)
  , m_lightData(this)
{
}

Group::~Group()
{
  cleanup();
}

bool Group::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      visionaraySceneConstruct();
      visionaraySceneCommit();
    }
    if (vscene && vscene->isValid()) {
      auto bounds = vscene->getBounds();
      std::memcpy(ptr, &bounds, sizeof(bounds));
      return true;
    }
  }

  return Object::getProperty(name, type, ptr, flags);
}

void Group::commitParameters()
{
  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");
  m_lightData = getParamObject<ObjectArray>("light");
}

void Group::finalize()
{
  cleanup();
}

void Group::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest =
      helium::newTimeStamp();
}

const std::vector<Surface *> &Group::surfaces() const
{
  return m_surfaces;
}

const std::vector<Volume *> &Group::volumes() const
{
  return m_volumes;
}

const std::vector<Light *> &Group::lights() const
{
  return m_lights;
}

VisionarayScene Group::visionarayScene() const
{
  return vscene;
}

void Group::visionaraySceneConstruct()
{
  const auto &state = *deviceState();
  if (m_objectUpdates.lastSceneConstruction
      > state.objectUpdates.lastBLSReconstructSceneRequest)
    return;

  reportMessage(
      ANARI_SEVERITY_DEBUG, "visionaray::Group rebuilding visionaray scene");

  if (vscene)
    vscene->release();
  vscene = newVisionarayScene(VisionaraySceneImpl::Group, deviceState());

  uint32_t id = 0;
  if (m_surfaceData) {
    std::for_each(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        [&](auto *o) {
          auto *s = (Surface *)o;
          if (s && s->isValid()) {
            m_surfaces.push_back(s);
            vscene->attachGeometry(s->geometry()->visionarayGeometry(),
                s->material()->visionarayMaterial(),
                id++, s->id());
          } else {
            reportMessage(ANARI_SEVERITY_DEBUG,
                "visionaray::Group rejecting invalid surface(%p) in building BLS",
                s);
            auto *g = s->geometry();
            if (!g || !g->isValid()) {
              reportMessage(
                  ANARI_SEVERITY_DEBUG, "    visionaray::Geometry is invalid");
            }
            auto *m = s->material();
            if (!m || !m->isValid()) {
              reportMessage(
                  ANARI_SEVERITY_DEBUG, "    visionaray::Material is invalid");
            }
          }
        });
  }

  uint32_t volID = 0;
  if (m_volumeData) {
    std::for_each(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        [&](auto *o) {
          auto *v = (Volume *)o;
          if (v && v->isValid()) {
            m_volumes.push_back(v);
            vscene->attachVolume(v->visionarayVolume(), volID++, v->id());
          } else {
            reportMessage(ANARI_SEVERITY_DEBUG,
                "visionaray::Group rejecting invalid volume(%p) in building BLS",
                v);
          }
        });
  }

  uint32_t lightID = 0;
  if (m_lightData) {
    std::for_each(m_lightData->handlesBegin(),
        m_lightData->handlesEnd(),
        [&](auto *o) {
          auto *l = (Light *)o;
          if (l && l->isValid()) {
            m_lights.push_back(l);
            vscene->attachLight(l->visionarayLight(), lightID++);
          } else {
            reportMessage(
                ANARI_SEVERITY_DEBUG, "    visionaray::Light is invalid");
          }
        });
  }

  m_objectUpdates.lastSceneConstruction = helium::newTimeStamp();
  m_objectUpdates.lastSceneCommit = 0;
  visionaraySceneCommit();
}

void Group::visionaraySceneCommit()
{
  const auto &state = *deviceState();
  if (m_objectUpdates.lastSceneCommit
      > state.objectUpdates.lastBLSCommitSceneRequest)
    return;

  reportMessage(
      ANARI_SEVERITY_DEBUG, "visionaray::Group committing visionaray scene");

  auto lock = scopeLockObject();
  vscene->commit();
  m_objectUpdates.lastSceneCommit = helium::newTimeStamp();
}

void Group::cleanup()
{
  m_surfaces.clear();
  m_volumes.clear();
  m_lights.clear();

  m_objectUpdates.lastSceneConstruction = 0;
  m_objectUpdates.lastSceneCommit = 0;

  if (vscene)
    vscene->release();
  vscene = nullptr;
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Group *);
