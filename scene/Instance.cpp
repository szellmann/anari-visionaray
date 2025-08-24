// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
// subtypes
#include "MotionTransform.h"

namespace visionaray {

Instance::Instance(VisionarayGlobalState *s)
  : Object(ANARI_INSTANCE, s)
  , m_xfmArray(this)
  , m_idArray(this)
{
  vinstance = dco::createInstance();
  vinstance.type = dco::Instance::Transform;
  vinstance.instID
      = deviceState()->dcos.instances.alloc(vinstance);

  m_xfms.resize(1);
  m_normalXfms.resize(1);
  m_affineInv.resize(1);
  m_transInv.resize(1);
}

Instance::~Instance()
{
  deviceState()->dcos.instances.free(vinstance.instID);
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

void Instance::commitParameters()
{
  m_idArray = getParamObject<Array1D>("id");
  m_id = getParam<uint32_t>("id", ~0u);
  m_xfmArray = getParamObject<Array1D>("transform");
  m_xfm = getParam<mat4>("transform", mat4::identity());
  m_group = getParamObject<Group>("group");

  float4 attrV(0.f, 0.f, 0.f, 1.f);
  if (getParam("attribute0", ANARI_FLOAT32_VEC4, &attrV))
    m_uniformAttributes[0] = attrV;
  if (getParam("attribute1", ANARI_FLOAT32_VEC4, &attrV))
    m_uniformAttributes[1] = attrV;
  if (getParam("attribute2", ANARI_FLOAT32_VEC4, &attrV))
    m_uniformAttributes[2] = attrV;
  if (getParam("attribute3", ANARI_FLOAT32_VEC4, &attrV))
    m_uniformAttributes[3] = attrV;
  if (getParam("color", ANARI_FLOAT32_VEC4, &attrV))
    m_uniformAttributes[4] = attrV;
}

void Instance::finalize()
{
  if (m_idArray && m_idArray->elementType() != ANARI_UINT32) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'id' array elements are %s, but need to be %s",
        anari::toString(m_idArray->elementType()),
        anari::toString(ANARI_UINT32));
    m_idArray = {};
  }
  if (m_xfmArray && m_xfmArray->elementType() != ANARI_FLOAT32_MAT4) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'transform' array elements are %s, but need to be %s",
        anari::toString(m_idArray->elementType()),
        anari::toString(ANARI_FLOAT32_MAT4));
    m_xfmArray = {};
  }
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");

  if (m_xfmArray) {
    m_xfms.clear();
    std::transform(m_xfmArray->beginAs<mat4>(),
        m_xfmArray->endAs<mat4>(),
        std::back_inserter(m_xfms),
        [](const mat4 &xfm) { return xfm; });
  } else {
    m_xfms[0] = m_xfm;
  }

  m_affineInv.clear();
  m_transInv.clear();
  m_normalXfms.clear();
  for (size_t i=0; i<m_xfms.size(); ++i) {
    m_affineInv.push_back(inverse(top_left(m_xfms[i])));
    m_transInv.push_back(-m_xfms[i](3).xyz());
    m_normalXfms.push_back(transpose(m_affineInv[i]));
  }

  for (int i=0; i<5; ++i) {
    if (m_uniformAttributes[i]) {
      vinstance.uniformAttributes[i].value = *m_uniformAttributes[i];
      vinstance.uniformAttributes[i].isSet = true;
    }
  }

  dispatch();
}

void Instance::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastTLSReconstructSceneRequest =
      helium::newTimeStamp();
}

bool Instance::isValid() const
{
  return m_group;
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

  for (int i=0; i<5; ++i) {
    if (m_uniformAttributes[i]) {
      vinstance.uniformAttributes[i].value = *m_uniformAttributes[i];
      vinstance.uniformAttributes[i].isSet = true;
    }
  }

  vinstance.theBVH = group()->visionarayScene()->refBVH();
  vinstance.xfms = m_xfms.devicePtr();
  vinstance.normalXfms = m_normalXfms.devicePtr();
  vinstance.affineInv = m_affineInv.devicePtr();
  vinstance.transInv = m_transInv.devicePtr();
  vinstance.len = m_xfms.size();
}

void Instance::dispatch()
{
  deviceState()->dcos.instances.update(vinstance.instID, vinstance);

  // Upload/set accessible pointers
  deviceState()->onDevice.instances = deviceState()->dcos.instances.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Instance *);
