#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
// nanovdb
#include <nanovdb/io/IO.h>
// ours
#include "NanoVDBField.h"

namespace visionaray {

NanoVDBField::NanoVDBField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::NanoVDB;
}

void NanoVDBField::commit()
{
  m_gridData = getParamObject<helium::Array1D>("gridData");

  if (!m_gridData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'gridHandle' on nanovdb spatial field");

    return;
  }

#ifdef WITH_CUDA
  cudaStream_t stream; // TODO: move to global state/use the one there?!
  cudaStreamCreate(&stream);
  nanovdb::cuda::DeviceBuffer buffer(m_gridData->totalSize(),
                                                    /*host:*/true,
                                                    &stream);
  memcpy(buffer.data(), m_gridData->data(), m_gridData->totalSize());
  m_gridHandle = std::move(buffer);

  m_gridHandle.deviceUpload(stream, false);

  vfield.asNanoVDB.grid = m_gridHandle.deviceGrid<float>();

  cudaStreamDestroy(stream);
#else
  auto buffer = nanovdb::HostBuffer::createFull(m_gridData->totalSize(),
                                                (void *)m_gridData->data());
  m_gridHandle = std::move(buffer);
  vfield.asNanoVDB.grid = m_gridHandle.grid<float>();
#endif

  vfield.voxelSpaceTransform = mat4x3(mat3::identity(),float3{0.f,0.f,0.f});

  dispatch();
}

bool NanoVDBField::isValid() const
{
  return (bool)m_gridHandle;
}

aabb NanoVDBField::bounds() const
{
  if (!isValid())
    return {};

  auto bbox = m_gridHandle.gridMetaData()->indexBBox();
  auto lower = bbox.min();
  auto upper = bbox.max();
  return aabb{{(float)lower[0], (float)lower[1], (float)lower[2]},
              {(float)upper[0], (float)upper[1], (float)upper[2]}};
}

} // namespace visionaray
