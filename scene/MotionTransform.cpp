// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "MotionTransform.h"

namespace visionaray {

MotionTransform::MotionTransform(VisionarayGlobalState *s)
  : Instance(s)
  , m_motionTransform(this)
{
  vinstance.type = dco::Instance::MotionTransform;
}

void MotionTransform::commitParameters()
{
  Instance::commitParameters();
  m_motionTransform = getParamObject<Array1D>("motion.transform");
  m_time = getParam<box1>("time", box1(0.f, 1.f));
}

void MotionTransform::finalize()
{
  if (!m_motionTransform) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'motion.transform' on motion transform instance");
    return;
  }

  m_xfms.resize(m_motionTransform->size());
  m_normalXfms.resize(m_motionTransform->size());
  m_affineInv.resize(m_motionTransform->size());
  m_transInv.resize(m_motionTransform->size());
  for (size_t i = 0; i < m_motionTransform->size(); ++i) {
    m_xfms[i] = m_motionTransform->dataAs<mat4>()[i];
    m_affineInv[i] = inverse(top_left(m_xfms[i]));
    m_transInv[i] = -m_xfms[i](3).xyz();
    m_normalXfms[i] = transpose(m_affineInv[i]);
  }

  Instance::dispatch();
}

void MotionTransform::visionarayInstanceUpdate()
{
  vinstance.userID = m_id;
  vinstance.groupID = group()->visionarayScene()->m_groupID;

  vinstance.theBVH = group()->visionarayScene()->refBVH();
  vinstance.xfms = m_xfms.devicePtr();
  vinstance.normalXfms = m_normalXfms.devicePtr();
  vinstance.affineInv = m_affineInv.devicePtr();
  vinstance.transInv = m_transInv.devicePtr();
  vinstance.len = m_xfms.size();
  vinstance.time = m_time;

  dispatch();
}

} // namespace visionaray
