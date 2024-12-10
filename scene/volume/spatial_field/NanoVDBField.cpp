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

  m_filter = getParamString("filter", "linear");

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

  vfield.asNanoVDB.filterMode = m_filter == "nearest" ? Nearest : Linear;

  vfield.voxelSpaceTransform = mat4x3(mat3::identity(),float3{0.f,0.f,0.f});

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

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

void NanoVDBField::buildGrid()
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return;
#endif
  int3 gridDims{16, 16, 16};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(gridDims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();
  auto acc = vfield.asNanoVDB.grid->getAccessor();

  float3 mcSize = (worldBounds.max-worldBounds.min) / float3(gridDims);

  for (unsigned mcz=0; mcz<gridDims.z; ++mcz) {
    for (unsigned mcy=0; mcy<gridDims.y; ++mcy) {
      for (unsigned mcx=0; mcx<gridDims.x; ++mcx) {
        const vec3i mcID(mcx,mcy,mcz);
        box3f mcBounds{worldBounds.min+mcSize*float3(mcx,mcy,mcz),
                       worldBounds.min+mcSize*float3(mcx+1,mcy+1,mcz+1)};

        nanovdb::CoordBBox bbox(
            {(int)mcBounds.min.x,(int)mcBounds.min.y,(int)mcBounds.min.z},
            {(int)mcBounds.max.x+1,(int)mcBounds.max.y+1,(int)mcBounds.max.z+1});

        for (nanovdb::CoordBBox::Iterator iter = bbox.begin(); iter; ++iter) {
          float value = acc.getValue(*iter);
          updateMC(mcID,gridDims,value,vaccel.valueRanges);
          updateMCStepSize(mcID,gridDims,0.5f,vaccel.stepSizes); // TODO!?
        }
      }
    }
  }
}

} // namespace visionaray
