
#include "Image1D.h"
#include "scene/surface/common.h"
#include "scene/surface/geometry/Geometry.h"

namespace visionaray {

Image1D::Image1D(VisionarayGlobalState *s) : Sampler(s)
{
  vsampler.type = dco::Sampler::Image1D;
}

bool Image1D::isValid() const
{
  return Sampler::isValid() && m_image;
}

void Image1D::commit()
{
  Sampler::commit();
  m_image = getParamObject<Array1D>("image");
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode = toAddressMode(getParamString("wrapMode1", "clampToEdge"));
  m_inTransform = getParam<mat4>("inTransform", mat4::identity());
  m_inOffset = getParam<float4>("inOffset", float4(0.f, 0.f, 0.f, 0.f));
  m_outTransform = getParam<mat4>("outTransform", mat4::identity());
  m_outOffset = getParam<float4>("outOffset", float4(0.f, 0.f, 0.f, 0.f));

  updateImageData();
  vsampler.inAttribute = m_inAttribute;
  vsampler.inTransform = m_inTransform;
  vsampler.inOffset = m_inOffset;
  vsampler.outTransform = m_outTransform;
  vsampler.outOffset = m_outOffset;
  vsampler.asImage1D = texture_ref<vector<4, unorm<8>>, 1>(vimage);

  Sampler::dispatch();
}

void Image1D::updateImageData()
{
  vimage = texture<vector<4, unorm<8>>, 1>(m_image->size());

  if (m_image->elementType() == ANARI_FLOAT32_VEC3)
    vimage.reset(m_image->dataAs<vec3>(), PF_RGB32F, PF_RGBA8, AlphaIsOne);
  else if (m_image->elementType() == ANARI_FLOAT32_VEC4)
    vimage.reset(m_image->dataAs<vec4>(), PF_RGBA32F, PF_RGBA8);
  else if (m_image->elementType() == ANARI_UFIXED8)
    vimage.reset((const unorm<8> *)m_image->data(), PF_R8, PF_RGBA8, AlphaIsOne);
  else if (m_image->elementType() == ANARI_UFIXED8_VEC3)
    vimage.reset((const vector<3, unorm<8>> *)m_image->data(),
                 PF_RGB8, PF_RGBA8, AlphaIsOne);
  else if (m_image->elementType() == ANARI_UFIXED8_VEC4)
    vimage.reset((const vector<4, unorm<8>> *)m_image->data());
  else if (m_image->elementType() == ANARI_UFIXED16_VEC3)
    vimage.reset((const vector<3, unorm<16>> *)m_image->data(),
                 PF_RGB16UI, PF_RGBA8, AlphaIsOne);
  else if (m_image->elementType() == ANARI_UFIXED16_VEC4)
    vimage.reset((const vector<4, unorm<16>> *)m_image->data(), PF_RGBA16UI, PF_RGBA8);
  else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unsupported element type Image1D sampler: %s",
        anari::toString(m_image->elementType()));
    return;
  }

  vimage.set_filter_mode(m_linearFilter?Linear:Nearest);
  vimage.set_address_mode(m_wrapMode);
}

} // namespace visionaray
