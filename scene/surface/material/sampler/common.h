
#pragma once

#include <visionaray/math/math.h>
#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"

namespace visionaray {

static bool imageSamplerFormatSupported(ANARIDataType type)
{
  return type == ANARI_FLOAT32_VEC3  ||
         type == ANARI_FLOAT32_VEC4  ||
         type == ANARI_UFIXED8       ||
         type == ANARI_UFIXED8_VEC3  ||
         type == ANARI_UFIXED8_VEC4  ||
         type == ANARI_UFIXED16_VEC4 ||
         type == ANARI_UFIXED16_VEC4;
}

template <typename Texture, typename Image>
inline bool imageSamplerUpdateData(Texture &tex, const Image &img)
{
  if (!imageSamplerFormatSupported(img->elementType()))
    return false;

  if (img->elementType() == ANARI_FLOAT32_VEC3)
    tex.reset(img->template dataAs<vec3>(), PF_RGB32F, PF_RGBA8, AlphaIsOne);
  else if (img->elementType() == ANARI_FLOAT32_VEC4)
    tex.reset(img->template dataAs<vec4>(), PF_RGBA32F, PF_RGBA8);
  else if (img->elementType() == ANARI_UFIXED8)
    tex.reset((const unorm<8> *)img->data(), PF_R8, PF_RGBA8, AlphaIsOne);
  else if (img->elementType() == ANARI_UFIXED8_VEC3)
    tex.reset((const vector<3, unorm<8>> *)img->data(),
              PF_RGB8, PF_RGBA8, AlphaIsOne);
  else if (img->elementType() == ANARI_UFIXED8_VEC4)
    tex.reset((const vector<4, unorm<8>> *)img->data());
  else if (img->elementType() == ANARI_UFIXED16_VEC3)
    tex.reset((const vector<3, unorm<16>> *)img->data(),
              PF_RGB16UI, PF_RGBA8, AlphaIsOne);
  else if (img->elementType() == ANARI_UFIXED16_VEC4)
    tex.reset((const vector<4, unorm<16>> *)img->data(), PF_RGBA16UI, PF_RGBA8);

  return true;
}

} // namespace visionaray
