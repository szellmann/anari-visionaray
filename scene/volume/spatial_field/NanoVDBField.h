
#pragma once

// nanovdb
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#ifdef WITH_CUDA
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
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
 private:

  helium::IntrusivePtr<Array1D> m_gridData;

  nanovdb::GridHandle<nanovdb::HostBuffer> m_gridHandle;
};

} // namespace visionaray
