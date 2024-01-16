
#include "BlockStructuredField.h"
#include "array/Array3D.h"

namespace visionaray {

BlockStructuredField::BlockStructuredField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::BlockStructured;
}

void BlockStructuredField::commit()
{
  m_params.cellWidth = getParamObject<helium::Array1D>("cellWidth");
  m_params.blockBounds = getParamObject<helium::Array1D>("block.bounds");
  m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
  m_params.blockData = getParamObject<helium::ObjectArray>("block.data");
  m_params.gridOrigin = getParam<float3>("gridOrigin", float3(0.f));

  if (!m_params.blockBounds) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.bounds' on amr spatial field");
    return;
  }

  if (!m_params.blockLevel) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.level' on amr spatial field");
    return;
  }

  if (!m_params.blockData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.data' on amr spatial field");
    return;
  }

  size_t numLevels = m_params.cellWidth->totalSize();
  size_t numBlocks = m_params.blockData->totalSize();
  auto *blockBounds = m_params.blockBounds->beginAs<aabbi>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *blockData = (Array3D **)m_params.blockData->handlesBegin();

  m_blocks.resize(numBlocks);

  std::vector<aabb> levelBounds(numLevels);
  for (auto &lb : levelBounds) {
    lb.invalidate();
  }

  for (size_t i=0; i<numBlocks; ++i) {
    m_blocks[i].bounds = blockBounds[i];
    m_blocks[i].level = blockLevels[i];
    m_blocks[i].scalarOffset = m_scalars.size();
    m_blocks[i].valueRange = box1f(FLT_MAX,-FLT_MAX);

    const Array3D *bd = *(blockData+i);

    for (unsigned z=0;z<bd->size().z;++z) {
      for (unsigned y=0;y<bd->size().y;++y) {
        for (unsigned x=0;x<bd->size().x;++x) {
          // TODO: can we actually iterate linearly here?!
          size_t index = z*size_t(bd->size().x)*bd->size().y 
                       + y*bd->size().x
                       + x;
          float f = bd->dataAs<float>()[index];
          m_scalars.push_back(f);
          m_blocks[i].valueRange.extend(f);
        }
      }
    }

    if (levelBounds.size() <= m_blocks[i].level) {
      levelBounds.resize(m_blocks[i].level + 1);
      levelBounds[m_blocks[i].level].invalidate();
    }
    levelBounds[m_blocks[i].level].insert(m_blocks[i].worldBounds());
  }

  aabb voxelBounds;
  voxelBounds.invalidate();
  m_bounds.invalidate();

  for (size_t i = 0; i < levelBounds.size(); ++i) {
    voxelBounds.insert(levelBounds[i]);

    float cw = m_params.cellWidth->dataAs<float>()[i];
    levelBounds[i].min *= cw;
    levelBounds[i].min += m_params.gridOrigin;
    levelBounds[i].max *= cw;
    levelBounds[i].max += m_params.gridOrigin;

    m_bounds.insert(levelBounds[i]);
  }

  // do this now that m_scalars doesn't change anymore:
  for (size_t i=0; i<numBlocks; ++i) {
    m_blocks[i].scalarsBuffer = m_scalars.devicePtr();
  }

  // sampling BVH

  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

#ifdef WITH_CUDA
  auto hostBVH = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  m_samplingBVH = cuda_index_bvh<dco::Block>(hostBVH);

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();
#else
  m_samplingBVH = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();
#endif

  mat3 S = mat3::scaling(voxelBounds.size()/m_bounds.size());
  vec3 T = voxelBounds.min-m_bounds.min;
  vfield.asBlockStructured.voxelSpaceTransform = mat4x3(S,T);

  setStepSize(length(voxelBounds.max-voxelBounds.min)/50.f);

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

  dispatch();
}

bool BlockStructuredField::isValid() const
{
  return m_samplingBVH.num_nodes();
}

aabb BlockStructuredField::bounds() const
{
  return m_bounds;
}

void BlockStructuredField::buildGrid()
{
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(dims, worldBounds);

  // TODO: not used, nor supported yet!
}

} // namespace visionaray
