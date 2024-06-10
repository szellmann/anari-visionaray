
#include "Image2D.h"
#include "common.h"
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
  vsampler.inTransform = m_inTransform;
  vsampler.inOffset = m_inOffset;
  vsampler.outTransform = m_outTransform;
  vsampler.outOffset = m_outOffset;
#ifdef WITH_CUDA
  vsampler.asImage2D = cuda_texture_ref<vector<4, unorm<8>>, 2>(vimage);
#else
  vsampler.asImage2D = texture_ref<vector<4, unorm<8>>, 2>(vimage);
#endif

  Sampler::dispatch();
}

void Image2D::updateImageData()
{
#ifdef WITH_CUDA
  texture<vector<4, unorm<8>>, 2> tex(m_image->size().x, m_image->size().y);
#else
  vimage = texture<vector<4, unorm<8>>, 2>(m_image->size().x, m_image->size().y);
  auto &tex = vimage;
#endif

  if (!imageSamplerUpdateData(tex, m_image)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unsupported element type Image2D sampler: %s",
        anari::toString(m_image->elementType()));
    return;
  }

  tex.set_filter_mode(m_linearFilter?Linear:Nearest);
  tex.set_address_mode(0, m_wrapMode1);
  tex.set_address_mode(1, m_wrapMode2);

#ifdef WITH_CUDA
  vimage = cuda_texture<vector<4, unorm<8>>, 2>(tex);
#endif
}

} // namespace visionaray
