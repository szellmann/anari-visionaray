#pragma once

#include "common.h"

namespace visionaray {

// ==================================================================
// conventional cdf sampling
// ==================================================================

VSNRAY_FUNC
inline const float* upper_bound(const float* first, const float* last, const float& val)
{
  const float* it;
  int count, step;
  count = (last-first);
  while (count > 0) {
    it = first; 
    step=count/2; 
    it += step;
    if (!(val < *it)) { 
      first=++it; 
      count-=step+1;  
    }
    else count=step;
  }
  return first;
}

struct CDFSample {
  unsigned x; // column
  unsigned y; // row
  float pdfx;
  float pdfy;
};

VSNRAY_FUNC
inline float sample_cdf(const float* data, unsigned int n, float x, unsigned int *idx, float* pdf) 
{
  *idx = upper_bound(data, data + n, x) - data;
  float scaled_sample;
  if (*idx == 0) {
    *pdf = data[0];
    scaled_sample = x / data[0];
  } else {
    if (*idx < n) {
    *pdf = data[*idx] - data[*idx - 1];
    scaled_sample = (x - data[*idx - 1]) / (data[*idx] - data[*idx - 1]);
    }
  }
  // keep result in [0,1)
  return min(scaled_sample, 0.99999994f);
}

// Uv range: [0, 1]
VSNRAY_FUNC
inline vec3f toPolar(vec2f uv)
{
  float theta = 2.0f * M_PI * uv.x + - M_PI / 2.0f;
  float phi = M_PI * uv.y;

  vec3f n;
  n.x = cos(theta) * sin(phi);
  n.z = sin(theta) * sin(phi);
  n.y = cos(phi);

  n.x = -n.x;
  return n;
}

VSNRAY_FUNC
inline vec2f toUV(vec3f n)
{
  vec2f uv;

  uv.x = atan2f(float(n.x), float(n.z));
  uv.x = (uv.x + M_PI / 2.0f) / (M_PI * 2.0f) + M_PI * (28.670f / 360.0f);

  uv.y = clamp(float(acosf(n.y) / M_PI), .001f, .999f);

  return uv;
}

VSNRAY_FUNC
inline CDFSample sampleCDF(const float *rows, const float *cols,
                           unsigned width, unsigned height, float rx, float ry)
{
  float row_pdf, col_pdf;
  unsigned x, y;
  sample_cdf(rows, height, ry, &y, &row_pdf);
  y = max(min(y, height - 1), 0u);
  sample_cdf(cols + y * width, width, rx, &x, &col_pdf);
  return {x,y,col_pdf,row_pdf};
}

} // namespace visionaray
