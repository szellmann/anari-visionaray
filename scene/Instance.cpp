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
  m_normalXfms[0] = inverse(transpose(m_affineInv[0]));

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

dco::Geometry Instance::visionarayGeometry() const
{
  return vgeom;
}

void Instance::visionarayGeometryUpdate()
{
  m_instance[0].userID = m_id;
  m_instance[0].groupID = group()->visionarayScene()->m_groupID;

  m_instance[0].theBVH = group()->visionarayScene()->refBVH();
  m_instance[0].xfms = m_xfms.devicePtr();
  m_instance[0].normalXfms = m_normalXfms.devicePtr();
  m_instance[0].affineInv = m_affineInv.devicePtr();
  m_instance[0].transInv = m_transInv.devicePtr();
  m_instance[0].len = m_xfms.size();

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
