
#pragma once

// nanovdb
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#ifdef WITH_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif
// ours
#include "SpatialField.h"
#include "array/Array1D.h"

namespace visionaray {

struct NanoVDBField : public SpatialField
{
  NanoVDBField(VisionarayGlobalState *d);
  ~NanoVDBField();

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;
 private:

  std::string m_filter;

  helium::IntrusivePtr<Array1D> m_gridData;

#ifdef WITH_CUDA
  nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> m_gridHandle;
#else
  nanovdb::GridHandle<nanovdb::HostBuffer> m_gridHandle;
  // Need to store an aligned copy of the data b/c helium
  // arrays aren't aligned
  // TODO: might move that functionality into HostArray
  // or similar
  float *m_gridDataAligned{nullptr};
#endif
};

} // namespace visionaray
