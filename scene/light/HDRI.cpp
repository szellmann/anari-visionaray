// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "HDRI.h"

namespace visionaray {

// Helper functions ///////////////////////////////////////////////////////////

template <typename InIt, typename OutIt>
void scan(InIt first, InIt last, OutIt dest, unsigned stride = 1)
{
  for (ptrdiff_t i = 0; i != last-first; i += stride) {
    *(dest + i) = i == 0 ? *first : *(dest + i - stride) + *(first + i);
  }
}

template <typename It>
void normalize(It first, It last)
{
  // Assumes that [first,last) is sorted!
  auto bck = *(last-1);
  if (bck != 0) {
    for (It it = first; it != last; ++it) {
      *it /= bck;
    }
  }
}

void makeCDF(const void *imgData, unsigned numComponents, int width, int height,
             HostDeviceArray<float> &cdfRows, HostDeviceArray<float> &cdfLastCol)
{
  // Build up luminance image
  std::vector<float> luminance(width * height);

  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      // That's a bit different than actual luminance,
      // but good enough here, and quite similar..
      if (numComponents == 3) {
        vec3 rgb = *((vec3 *)imgData + y * width + x);
        luminance[y*width+x] = max(max(rgb.x,rgb.y),rgb.z);
      } else if (numComponents == 4) {
        vec4 rgba = *((vec4 *)imgData + y * width + x);
        luminance[y*width+x] = max(max(max(rgba.x,rgba.y),rgba.z),rgba.w);
      }
    }
  }

  // Build up CDF
  cdfRows.resize(width * height, 0.f);
  cdfLastCol.resize(height);
  std::vector<float> lastCol(height);

  for (int y=0; y<height; ++y) {
    // Scan each row
    size_t off = y * width;
    scan(luminance.data() + off, luminance.data() + off + width, cdfRows.data() + off);
    // Assemble the last column by filling with the last item of each row
    lastCol[y] = *(cdfRows.data() + off + width - 1);
    // Normalize the row
    normalize(cdfRows.data() + off, cdfRows.data() + off + width);
  }

  // Scan and normalize the last column
  scan(lastCol.begin(), lastCol.end(), cdfLastCol.begin());
  normalize(cdfLastCol.data(), cdfLastCol.data()+cdfLastCol.size());
}

void makeRGBA(const void *imgData, unsigned numComponents, int width, int height,
              std::vector<float4> &rgba)
{
  rgba.resize(width * height);
  if (numComponents == 3) {
    auto *rgbData = (const float3 *)imgData;
    for (int y=0; y<height; ++y) {
      for (int x=0; x<width; ++x) {
        float3 rgb = rgbData[x+width*y];
        rgba[x+width*y] = float4(rgb,1.f);
      }
    }
  } else if (numComponents == 4) {
    memcpy(rgba.data(), imgData, width*height*sizeof(float4));
  } else {
    assert(0);
  }
}

// HDRI definitions ///////////////////////////////////////////////////////////

int HDRI::backgroundID = -1;

HDRI::HDRI(VisionarayGlobalState *s) : Light(s)
{
  vlight.type = dco::Light::HDRI;
}

HDRI::~HDRI()
{
  if (m_visible)
    backgroundID = -1; // reset!
}

void HDRI::commitParameters()
{
  Light::commitParameters();
  m_up = getParam<vec3>("up", vec3(1.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, 1.f));
  m_scale = getParam<float>("scale", 1.f);

  m_radiance = getParamObject<Array2D>("radiance");
}

void HDRI::finalize()
{
  Light::finalize();

  if (!m_radiance) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'radiance' on 'hdri' light");
    return;
  }

  unsigned width = m_radiance->size().x, height = m_radiance->size().y;
  makeCDF(m_radiance->data(), 3, width, height, m_cdfRows, m_cdfLastCol);

  std::vector<float4> rgba; // convert to rgba for compability with CUDA
                            // (no support for float3 textures!)
  makeRGBA(m_radiance->data(), 3, width, height, rgba);

#if defined(WITH_CUDA) || defined(WITH_HIP)
  texture<float4, 2> tex(width, height);
#else
  m_radianceTexture = texture<float4, 2>(width, height);
  auto &tex = m_radianceTexture;
#endif
  tex.reset((const float4 *)rgba.data());
  tex.set_filter_mode(Linear);
  tex.set_address_mode(Clamp);

#ifdef WITH_CUDA
  m_radianceTexture = cuda_texture<float4, 2>(tex);
#elif defined(WITH_HIP)
  m_radianceTexture = hip_texture<float4, 2>(tex);
#endif

#ifdef WITH_CUDA
  vlight.asHDRI.radiance = cuda_texture_ref<float4, 2>(m_radianceTexture);
#elif defined(WITH_HIP)
  vlight.asHDRI.radiance = hip_texture_ref<float4, 2>(m_radianceTexture);
#else
  vlight.asHDRI.radiance = texture_ref<float4, 2>(m_radianceTexture);
#endif
  vlight.asHDRI.scale = m_scale;
  vlight.asHDRI.toWorld.col1 = -normalize(m_up);
  vlight.asHDRI.toWorld.col0 =  normalize(cross(vlight.asHDRI.toWorld.col1,m_direction));
  vlight.asHDRI.toWorld.col2 =  normalize(cross(vlight.asHDRI.toWorld.col0,
                                                vlight.asHDRI.toWorld.col1));
  vlight.asHDRI.toLocal = inverse(vlight.asHDRI.toWorld);
  vlight.asHDRI.cdf.lastCol = m_cdfRows.devicePtr();
  vlight.asHDRI.cdf.rows = m_cdfLastCol.devicePtr();
  vlight.asHDRI.cdf.width = width;
  vlight.asHDRI.cdf.height = height;

  if (m_visible) {
    backgroundID = vlight.lightID;
  } else {
    if (backgroundID == vlight.lightID)
      backgroundID = -1;
  }

  dispatch();
}

} // visionaray
