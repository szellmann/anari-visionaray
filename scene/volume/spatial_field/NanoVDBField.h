
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

  void commit() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;
 private:

  helium::IntrusivePtr<Array1D> m_gridData;

#ifdef WITH_CUDA
  nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> m_gridHandle;
#else
  nanovdb::GridHandle<nanovdb::HostBuffer> m_gridHandle;
#endif
};

} // namespace visionaray
