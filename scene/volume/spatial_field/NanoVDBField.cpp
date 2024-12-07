// nanovdb
#include <nanovdb/util/IO.h>
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

  auto buffer = nanovdb::HostBuffer::createFull(m_gridData->totalSize(),
                                                (void *)m_gridData->data());
  m_gridHandle = std::move(buffer);

  vfield.asNanoVDB.grid = m_gridHandle.grid<float>();

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
  return aabb{{lower[0], lower[1], lower[2]},
              {upper[0], upper[1], upper[2]}};
}

} // namespace visionaray
