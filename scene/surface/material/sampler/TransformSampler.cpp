
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

void TransformSampler::commitParameters()
{
  Sampler::commitParameters();
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_outTransform = getParam<mat4>("transform", mat4::identity());
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform); // new variant!
  m_outOffset = getParam<float4>("outOffset", float4(0.f, 0.f, 0.f, 0.f));
}

void TransformSampler::finalize()
{
  vsampler.inAttribute = m_inAttribute;
  vsampler.outTransform = m_outTransform;
  vsampler.outOffset = m_outOffset;

  Sampler::dispatch();
}

} // namespace visionaray
