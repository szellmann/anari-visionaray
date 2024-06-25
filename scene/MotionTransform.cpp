#include "MotionTransform.h"

namespace visionaray {

MotionTransform::MotionTransform(VisionarayGlobalState *s)
  : Instance(s)
  , m_motionTransform(this)
{
  m_instance[0].type = dco::Instance::MotionTransform;
}

void MotionTransform::commit()
{
  Instance::commit();

  m_motionTransform = getParamObject<Array1D>("motion.transform");
  m_time = getParam<box1>("time", box1(0.f, 1.f));

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
    m_affineInv[i] = inverse(transpose(top_left(m_xfms[i])));
    m_transInv[i] = -m_xfms[i](3).xyz();
    m_normalXfms[i] = inverse(m_affineInv[i]);
  }

  Instance::dispatch();
}

void MotionTransform::visionarayGeometryUpdate()
{
  m_instance[0].userID = m_id;
  m_instance[0].groupID = group()->visionarayScene()->m_groupID;

  m_instance[0].theBVH = group()->visionarayScene()->refBVH();
  m_instance[0].xfms = m_xfms.devicePtr();
  m_instance[0].normalXfms = m_normalXfms.devicePtr();
  m_instance[0].affineInv = m_affineInv.devicePtr();
  m_instance[0].transInv = m_transInv.devicePtr();
  m_instance[0].len = m_xfms.size();
  m_instance[0].time = m_time;

  vgeom.primitives.data = m_instance.devicePtr();
  vgeom.primitives.len = m_instance.size();

  dispatch();
}

} // namespace visionaray
