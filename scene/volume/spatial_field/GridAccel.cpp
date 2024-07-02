#include "for_each.h"
#include "GridAccel.h"

namespace visionaray {

GridAccel::GridAccel(VisionarayGlobalState *s) : m_state(s) {}

void GridAccel::init(int3 dims, box3 worldBounds)
{
  m_dims = dims;
  m_worldBounds = worldBounds;

  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

  delete[] m_valueRanges;
  delete[] m_maxOpacities;

  m_valueRanges = new box1[numMCs];
  m_maxOpacities = new float[numMCs];

  for (size_t i=0; i<numMCs; ++i) {
    auto &vr = m_valueRanges[i];
    vr = {FLT_MAX, -FLT_MAX};
  }

  vaccel.dims = m_dims;
  vaccel.worldBounds = m_worldBounds;
  vaccel.valueRanges = m_valueRanges;
  vaccel.maxOpacities = m_maxOpacities;
}

void GridAccel::cleanup()
{
  delete[] m_valueRanges;
  delete[] m_maxOpacities;
}

dco::GridAccel &GridAccel::visionarayAccel()
{
  return vaccel;
}

box1 *GridAccel::valueRanges()
{
  return m_valueRanges;
}

VisionarayGlobalState *GridAccel::deviceState() const
{
  return m_state;
}

void GridAccel::computeMaxOpacities(dco::TransferFunction1D tf)
{
  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

  parallel::for_each(deviceState()->threadPool, 0, numMCs,
    [&](size_t threadID) {
      box1 valueRange = m_valueRanges[threadID];

      if (valueRange.max < valueRange.min) {
        m_maxOpacities[threadID] = 0.f;
        return;
      }

      valueRange.min -= tf.valueRange.min;
      valueRange.min /= tf.valueRange.max - tf.valueRange.min;
      valueRange.max -= tf.valueRange.min;
      valueRange.max /= tf.valueRange.max - tf.valueRange.min;

      int numValues = tf.numValues;

      int lo = clamp(
          int(valueRange.min * (numValues - 1)), 0, numValues - 1);
      int hi = clamp(
          int(valueRange.max * (numValues - 1)) + 1, 0, numValues - 1);

      float maxOpacity = 0.f;
      for (int i = lo; i <= hi; ++i) {
        float tc = (i + .5f) / numValues;
        maxOpacity = fmaxf(maxOpacity, tex1D(tf.sampler, tc).w);
      }
      m_maxOpacities[threadID] = maxOpacity;
    });
}

} // namespace visionaray
