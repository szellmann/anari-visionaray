
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

  size_t numBlocks = m_params.blockData->totalSize();
  auto *blockBounds = m_params.blockBounds->beginAs<aabbi>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *blockData = (Array3D **)m_params.blockData->handlesBegin();

  m_blocks.resize(numBlocks);

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
  }

  // do this now that m_scalars doesn't change anymore:
  for (size_t i=0; i<numBlocks; ++i) {
    m_blocks[i].scalarsBuffer = m_scalars.data();
  }

  // sampling BVH

  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

  m_samplingBVH = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();

  setStepSize(length(bounds().max-bounds().min)/50.f);

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
  if (isValid())
    return m_samplingBVH.node(0).get_bounds();

  return {};
}

void BlockStructuredField::buildGrid()
{
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(dims, worldBounds);

  // TODO: not used, nor supported yet!
}

} // namespace visionaray
