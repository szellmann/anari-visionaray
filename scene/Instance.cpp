// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
// subtypes
#include "MotionTransform.h"

namespace visionaray {

Instance::Instance(VisionarayGlobalState *s) : Object(ANARI_INSTANCE, s)
{
  vgeom.type = dco::Geometry::Instance;
  vgeom.geomID = deviceState()->dcos.geometries.alloc(vgeom);
  m_instance.resize(1);
  m_instance[0].type = dco::Instance::Transform;
  m_instance[0].instID
      = deviceState()->dcos.instances.alloc(m_instance[0]);
  s->objectCounts.instances++;
}

Instance::~Instance()
{
  deviceState()->dcos.instances.free(m_instance[0].instID);
  deviceState()->dcos.geometries.free(vgeom.geomID);

  deviceState()->objectCounts.instances--;
}

Instance *Instance::createInstance(
    std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "transform")
    return new Instance(s); // base type implements transform!
  else if (subtype == "motionTransform")
    return new MotionTransform(s);
  else
    return new Instance(s); // base type implements transform!
}

void Instance::commit()
{
  m_id = getParam<uint32_t>("id", ~0u);
  m_xfm = getParam<mat4>("transform", mat4::identity());
  m_xfmInvRot = inverse(top_left(m_xfm));
  m_group = getParamObject<Group>("group");
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");

  dispatch();
}

uint32_t Instance::id() const
{
  return m_id;
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
  m_instance[0].userID = m_id;
  m_instance[0].groupID = group()->visionarayScene()->m_groupID;

  // set xfm
  m_instance[0].theBVH = group()->visionarayScene()->refBVH();
  m_instance[0].asTransform.xfm = m_xfm;
  m_instance[0].asTransform.affineInv = inverse(top_left(m_xfm));
  m_instance[0].asTransform.transInv = -m_xfm(3).xyz();
  m_instance[0].asTransform.normalXfm
      = inverse(transpose(m_instance[0].asTransform.affineInv ));

  vgeom.primitives.data = m_instance.devicePtr();
  vgeom.primitives.len = m_instance.size();

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
  deviceState()->dcos.instances.update(m_instance[0].instID, m_instance[0]);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Instance *);
