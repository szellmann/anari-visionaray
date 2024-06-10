
#include "Image1D.h"
#include "common.h"
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
#ifdef WITH_CUDA
  vsampler.asImage1D = cuda_texture_ref<vector<4, unorm<8>>, 1>(vimage);
#else
  vsampler.asImage1D = texture_ref<vector<4, unorm<8>>, 1>(vimage);
#endif

  Sampler::dispatch();
}

void Image1D::updateImageData()
{
#ifdef WITH_CUDA
  texture<vector<4, unorm<8>>, 1> tex(m_image->size());
#else
  vimage = texture<vector<4, unorm<8>>, 1>(m_image->size());
  auto &tex = vimage;
#endif

  if (!imageSamplerUpdateData(tex, m_image)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unsupported element type Image1D sampler: %s",
        anari::toString(m_image->elementType()));
    return;
  }

  tex.set_filter_mode(m_linearFilter?Linear:Nearest);
  tex.set_address_mode(m_wrapMode);

#ifdef WITH_CUDA
  vimage = cuda_texture<vector<4, unorm<8>>, 1>(tex);
#endif
}

} // namespace visionaray
