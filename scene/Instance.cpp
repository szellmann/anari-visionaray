// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
// subtypes
#include "MotionTransform.h"

namespace visionaray {

Instance::Instance(VisionarayGlobalState *s) : Object(ANARI_INSTANCE, s)
{
  vinstance.type = dco::Instance::Transform;
  vinstance.instID
      = deviceState()->dcos.instances.alloc(vinstance);
  s->objectCounts.instances++;
}

Instance::~Instance()
{
  deviceState()->dcos.instances.free(vinstance.instID);

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
  m_group = getParamObject<Group>("group");
  mat4 xfm = getParam<mat4>("transform", mat4::identity());

  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");

  m_xfms.resize(1);
  m_normalXfms.resize(1);
  m_affineInv.resize(1);
  m_transInv.resize(1);

  m_xfms[0] = xfm;
  m_affineInv[0] = inverse(top_left(m_xfms[0]));
  m_transInv[0] = -m_xfms[0](3).xyz();
  m_normalXfms[0] = transpose(m_affineInv[0]);

  dispatch();
}

uint32_t Instance::id() const
{
  return m_id;
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

Group *Instance::group()
{
  return m_group.ptr;
}

dco::Instance Instance::visionarayInstance() const
{
  return vinstance;
}

void Instance::visionarayInstanceUpdate()
{
  vinstance.userID = m_id;
  vinstance.groupID = group()->visionarayScene()->m_groupID;

  vinstance.theBVH = group()->visionarayScene()->refBVH();
  vinstance.xfms = m_xfms.devicePtr();
  vinstance.normalXfms = m_normalXfms.devicePtr();
  vinstance.affineInv = m_affineInv.devicePtr();
  vinstance.transInv = m_transInv.devicePtr();
  vinstance.len = m_xfms.size();

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
  deviceState()->dcos.instances.update(vinstance.instID, vinstance);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Instance *);
