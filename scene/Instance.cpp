// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"

namespace visionaray {

Instance::Instance(VisionarayGlobalState *s) : Object(ANARI_INSTANCE, s)
{
  vgeom.type = dco::Geometry::Instance;
  vgeom.asInstance.data.instID
      = deviceState()->dcos.instances.alloc(vgeom.asInstance.data);
  s->objectCounts.instances++;
}

Instance::~Instance()
{
  detach();

  // rtcReleaseGeometry(m_embreeGeometry);
  deviceState()->objectCounts.instances--;
}

void Instance::commit()
{
  m_xfm = getParam<mat4>("transform", mat4::identity());
  m_xfmInvRot = inverse(top_left(m_xfm));
  m_group = getParamObject<Group>("group");
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");

  dispatch();
}

const mat4 &Instance::xfm() const
{
  return m_xfm;
}

const mat3 &Instance::xfmInvRot() const
{
  return m_xfmInvRot;
}

bool Instance::xfmIsIdentity() const
{
  return xfm() == mat4::identity();
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

Group *Instance::group()
{
  return m_group.ptr;
}

dco::Geometry Instance::visionarayGeometry() const
{
  return vgeom;
}

void Instance::visionarayGeometryUpdate()
{
  // rtcSetGeometryInstancedScene(m_embreeGeometry, group()->embreeScene());
  vgeom.asInstance.data.scene = group()->visionarayScene();
  vgeom.asInstance.data.groupID = group()->visionarayScene()->m_groupID;
  vgeom.asInstance.data.xfm = m_xfm;
  // rtcSetGeometryTransform(
  //     m_embreeGeometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &m_xfm);
  // rtcCommitGeometry(m_embreeGeometry);
  dispatch();
}

void Instance::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastTLSReconstructSceneRequest =
      helium::newTimeStamp();
}

bool Instance::isValid() const
{
  return m_group;
}

void Instance::dispatch()
{
  vgeom.asInstance.data.invXfm = inverse(vgeom.asInstance.data.xfm);
  deviceState()->dcos.instances.update(
      vgeom.asInstance.data.instID, vgeom.asInstance.data);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

void Instance::detach()
{
  deviceState()->dcos.instances.free(vgeom.asInstance.data.instID);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Instance *);
