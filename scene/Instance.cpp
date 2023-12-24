// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"

namespace visionaray {

Instance::Instance(VisionarayGlobalState *s) : Object(ANARI_INSTANCE, s)
{
  vgeom.type = dco::Geometry::Instance;
  vgeom.geomID = deviceState()->dcos.geometries.alloc(vgeom);
  vgeom.asInstance.data.instID
      = deviceState()->dcos.instances.alloc(vgeom.asInstance.data);
  s->objectCounts.instances++;
}

Instance::~Instance()
{
  deviceState()->dcos.instances.free(vgeom.asInstance.data.instID);
  deviceState()->dcos.geometries.free(vgeom.geomID);

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
  vgeom.asInstance.data.groupID = group()->visionarayScene()->m_groupID;
  vgeom.asInstance.data.xfm = m_xfm;

  // set xfm
  mat3f rot = top_left(vgeom.asInstance.data.xfm);
  vec3f trans(vgeom.asInstance.data.xfm(0,3),
              vgeom.asInstance.data.xfm(1,3),
              vgeom.asInstance.data.xfm(2,3));
  mat4x3 xfm{rot, trans};
  vgeom.asInstance.data.instBVH = group()->visionarayScene()->instBVH(xfm);

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
  deviceState()->dcos.geometries.update(vgeom.geomID, vgeom);
  vgeom.asInstance.data.invXfm = inverse(vgeom.asInstance.data.xfm);
  deviceState()->dcos.instances.update(
      vgeom.asInstance.data.instID, vgeom.asInstance.data);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Instance *);
