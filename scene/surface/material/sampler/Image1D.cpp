
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
  vsampler.asImage1D = texture_ref<vector<4, unorm<8>>, 1>(vimage);

  Sampler::dispatch();
}

void Image1D::updateImageData()
{
  // TODO: generalize for other image sampler types!
  using InternalType = vector<4, unorm<8>>;
  std::vector<InternalType> data(m_image->size());
  if (m_image->elementType() == ANARI_FLOAT32_VEC3) {
    auto *in = m_image->beginAs<vec3>();
    for (size_t i = 0; i < m_image->size(); ++i) {
      data[i] = InternalType(in[i].x, in[i].y, in[i].z, 1.f);
    }
  }

  vimage = texture<InternalType, 1>(data.size());
  vimage.reset(data.data());
  vimage.set_filter_mode(m_linearFilter?Linear:Nearest);
  vimage.set_address_mode(m_wrapMode);
}

} // namespace visionaray
