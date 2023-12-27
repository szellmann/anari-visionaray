
#include "scene/surface/common.h"
#include "TransformSampler.h"

namespace visionaray {

TransformSampler::TransformSampler(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Transform;
}

bool TransformSampler::isValid() const
{
  return Sampler::isValid();
}

void TransformSampler::commit()
{
  Sampler::commit();
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_transform = getParam<mat4>("transform", mat4::identity());

  vsampler.inAttribute = m_inAttribute;
  vsampler.asTransform = m_transform;

  Sampler::dispatch();
}

} // namespace visionaray
