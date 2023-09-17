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
             aligned_vector<float> &cdfRows, aligned_vector<float> &cdfLastCol)
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
  normalize(cdfLastCol.begin(), cdfLastCol.end());
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

  detach();
}

void HDRI::commit()
{
  Light::commit();
  m_up = getParam<vec3>("up", vec3(1.f, 0.f, 0.f));
  m_direction = getParam<vec3>("direction", vec3(0.f, 0.f, 1.f));
  m_scale = getParam<float>("scale", 1.f);

  m_radiance = getParamObject<Array2D>("radiance");

  if (!m_radiance) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'radiance' on 'hdri' light");
    return;
  }

  unsigned width = m_radiance->size().x, height = m_radiance->size().y;
  makeCDF(m_radiance->data(), 3, width, height, m_cdfRows, m_cdfLastCol);

  m_radianceTexture = texture<float3, 2>(width, height);
  m_radianceTexture.reset((const float3 *)m_radiance->data());
  m_radianceTexture.set_filter_mode(Linear);
  m_radianceTexture.set_address_mode(Clamp);

  vlight.asHDRI.radiance = texture_ref<float3, 2>(m_radianceTexture);
  vlight.asHDRI.scale = m_scale;
  vlight.asHDRI.cdf.lastCol = m_cdfRows.data();
  vlight.asHDRI.cdf.rows = m_cdfLastCol.data();
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
