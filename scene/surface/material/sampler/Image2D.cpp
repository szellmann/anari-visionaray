
#include "Image2D.h"
#include "scene/surface/common.h"
#include "scene/surface/geometry/Geometry.h"

namespace visionaray {

Image2D::Image2D(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Image2D;
}

bool Image2D::isValid() const
{
  return Sampler::isValid() && m_image;
}

void Image2D::commit()
{
  Sampler::commit();
  m_image = getParamObject<Array2D>("image");
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode1 = toAddressMode(getParamString("wrapMode1", "clampToEdge"));
  m_wrapMode2 = toAddressMode(getParamString("wrapMode2", "clampToEdge"));
  m_inTransform = getParam<mat4>("inTransform", mat4::identity());
  m_inOffset = getParam<float4>("inOffset", float4(0.f, 0.f, 0.f, 0.f));
  m_outTransform = getParam<mat4>("outTransform", mat4::identity());
  m_outOffset = getParam<float4>("outOffset", float4(0.f, 0.f, 0.f, 0.f));

  updateImageData();
  vsampler.inAttribute = m_inAttribute;
  vsampler.inAttribute = m_inAttribute;
  vsampler.inTransform = m_inTransform;
  vsampler.inOffset = m_inOffset;
  vsampler.outTransform = m_outTransform;
  vsampler.outOffset = m_outOffset;
  vsampler.asImage2D = texture_ref<vector<4, unorm<8>>, 2>(vimage);

  Sampler::dispatch();
}

void Image2D::updateImageData()
{
  vimage = texture<vector<4, unorm<8>>, 2>(m_image->size().x, m_image->size().y);

  if (m_image->elementType() == ANARI_FLOAT32_VEC3)
    vimage.reset(m_image->dataAs<vec3>(), PF_RGB32F, PF_RGBA8, AlphaIsOne);
  else if (m_image->elementType() == ANARI_FLOAT32_VEC4)
    vimage.reset(m_image->dataAs<vec4>(), PF_RGBA32F, PF_RGBA8);

  vimage.set_filter_mode(m_linearFilter?Linear:Nearest);
  vimage.set_address_mode(0, m_wrapMode1);
  vimage.set_address_mode(1, m_wrapMode2);
}

} // namespace visionaray
