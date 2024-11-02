#include "for_each.h"
#include "GridAccel.h"

namespace visionaray {

GridAccel::GridAccel(VisionarayGlobalState *s) : m_state(s) {}

void GridAccel::init(int3 dims, box3 worldBounds)
{
  m_dims = dims;
  m_worldBounds = worldBounds;

  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

  m_stepSizes.resize(numMCs);
  m_valueRanges.resize(numMCs);
  m_maxOpacities.resize(numMCs);

  for (size_t i=0; i<numMCs; ++i) {
    m_stepSizes[i] = 1e30f;
    m_valueRanges[i] = {FLT_MAX, -FLT_MAX};
  }

  vaccel.dims = m_dims;
  vaccel.worldBounds = m_worldBounds;
  vaccel.stepSizes = m_stepSizes.devicePtr();
  vaccel.valueRanges = m_valueRanges.devicePtr();
  vaccel.maxOpacities = m_maxOpacities.devicePtr();
}

dco::GridAccel &GridAccel::visionarayAccel()
{
  return vaccel;
}

bool GridAccel::isValid() const
{
  return vaccel.isValid();
}

VisionarayGlobalState *GridAccel::deviceState() const
{
  return m_state;
}

void GridAccel::computeMaxOpacities(dco::TransferFunction1D tf)
{
  size_t numMCs = m_dims.x * size_t(m_dims.y) * m_dims.z;

#ifdef WITH_CUDA
  dco::GridAccel *gridPtr;
  CUDA_SAFE_CALL(cudaMalloc(&gridPtr, sizeof(vaccel)));
  CUDA_SAFE_CALL(cudaMemcpy(gridPtr, &vaccel, sizeof(vaccel), cudaMemcpyHostToDevice));
  cuda::for_each(0, numMCs,
#else
  auto *gridPtr = &vaccel;
  parallel::for_each(deviceState()->threadPool, 0, numMCs,
#endif
    [=] VSNRAY_GPU_FUNC (size_t threadID) {
      const auto &vaccel = *gridPtr;

      box1 valueRange = vaccel.valueRanges[threadID];

      if (valueRange.max < valueRange.min) {
        vaccel.maxOpacities[threadID] = 0.f;
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
      vaccel.maxOpacities[threadID] = maxOpacity;
    });
#ifdef WITH_CUDA
  CUDA_SAFE_CALL(cudaFree(gridPtr));
#endif
}

} // namespace visionaray
