
#include "BlockStructuredField.h"
#include "for_each.h"
#include "array/Array3D.h"

namespace visionaray {

BlockStructuredField::BlockStructuredField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::BlockStructured;
}

void BlockStructuredField::commitParameters()
{
  m_params.refinementRatio = getParamObject<helium::Array1D>("refinementRatio");
  m_params.blockBounds = getParamObject<helium::Array1D>("block.bounds");
  m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
  m_params.data = getParamObject<helium::Array1D>("data");
  m_params.gridOrigin = getParam<float3>("gridOrigin", float3(0.f));
  m_params.gridSpacing = getParam<float3>("gridSpacing", float3(1.f));
}

void BlockStructuredField::finalize()
{
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

  if (!m_params.data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on amr spatial field");
    return;
  }

  size_t numLevels = m_params.refinementRatio->totalSize();
  size_t numBlocks = m_params.blockBounds->totalSize();
  auto *refinementRatio = m_params.refinementRatio->beginAs<unsigned>();
  auto *blockBounds = m_params.blockBounds->beginAs<aabbi>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *data = m_params.data->beginAs<float>();

  m_blocks.resize(numBlocks);

  std::vector<aabb> levelBounds(numLevels);
  for (auto &lb : levelBounds) {
    lb.invalidate();
  }

  for (size_t i=0; i<numBlocks; ++i) {
    int3 blockSize = blockBounds[i].max - blockBounds[i].min + int3(1);

    m_blocks[i].bounds = blockBounds[i];
    m_blocks[i].level = blockLevels[i];
    m_blocks[i].scalarOffset = m_scalars.size();
    m_blocks[i].valueRange = box1f(FLT_MAX,-FLT_MAX);

    float cellWidth = powf((float)refinementRatio[blockLevels[i]], (float)blockLevels[i]);
    (void)cellWidth; // ignore for now

    for (int j=0; j<blockSize.x*blockSize.y*blockSize.z; ++j) {
      size_t index = size_t(m_blocks[i].scalarOffset) + j;
      float f = data[index];
      m_scalars.push_back(f);
      m_blocks[i].valueRange.extend(f);
    }

    if (levelBounds.size() <= m_blocks[i].level) {
      levelBounds.resize(m_blocks[i].level + 1);
      levelBounds[m_blocks[i].level].invalidate();
    }
    levelBounds[m_blocks[i].level].insert(m_blocks[i].worldBounds());
  }

  aabb voxelBounds;
  voxelBounds.invalidate();
  for (size_t i = 0; i < levelBounds.size(); ++i) {
    voxelBounds.insert(levelBounds[i]);
  }
  m_bounds.min = m_params.gridOrigin + voxelBounds.min * m_params.gridSpacing;
  m_bounds.max = m_params.gridOrigin + voxelBounds.max * m_params.gridSpacing;

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
  auto samplingBVH2 = builder.build(
    index_bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

  bvh_collapser collapser;
  collapser.collapse(samplingBVH2, m_samplingBVH, deviceState()->threadPool);

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();
#endif

  mat3 S = mat3::scaling(voxelBounds.size()/m_bounds.size());
  vec3 T = voxelBounds.min-m_bounds.min;
  vfield.voxelSpaceTransform = mat4x3(S,T);

  setCellSize(1.f);

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

#ifdef WITH_CUDA
__global__ void BlockStructuredField_buildGridGPU(dco::GridAccel    vaccel,
                                                  const dco::Block *blocks,
                                                  size_t            numBlocks)
{
  size_t blockID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  if (blockID >= numBlocks)
    return;

  const auto &block = blocks[blockID];
  int cellSize = block.cellSize();

  for (int z=0; z<block.numCells().z; ++z) {
    for (int y=0; y<block.numCells().y; ++y) {
      for (int x=0; x<block.numCells().x; ++x) {
        vec3i cellID(x,y,z);
        vec3i cell_lower = (block.bounds.min+cellID)*cellSize;
        vec3i cell_upper = (block.bounds.min+cellID+vec3i(1))*cellSize;
        aabb cellBounds(vec3f(cell_lower)-vec3f(cellSize*0.5f),
                        vec3f(cell_upper)+vec3f(cellSize*0.5f)); // +/- filterDomain
        float scalar = block.getScalar(x,y,z);

        const vec3i loMC = projectOnGrid(cellBounds.min,vaccel.dims,vaccel.worldBounds);
        const vec3i upMC = projectOnGrid(cellBounds.max,vaccel.dims,vaccel.worldBounds);

        for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
          for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
            for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
              const vec3i mcID(mcx,mcy,mcz);
              updateMC(mcID,vaccel.dims,scalar,vaccel.valueRanges);
              updateMCStepSize(mcID,vaccel.dims,cellSize,vaccel.stepSizes);
            }
          }
        }
      }
    }
  }
}
#endif

void BlockStructuredField::buildGrid()
{
#ifdef WITH_CUDA
  box3f worldBounds = {bounds().min,bounds().max};
  worldBounds.min = vfield.pointToVoxelSpace(worldBounds.min);
  worldBounds.max = vfield.pointToVoxelSpace(worldBounds.max);
  int3 dims{
    div_up(int(worldBounds.max.x-worldBounds.min.x),8),
    div_up(int(worldBounds.max.y-worldBounds.min.y),8),
    div_up(int(worldBounds.max.z-worldBounds.min.z),8)
  };
  m_gridAccel.init(dims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numThreads = 1024;
  size_t numBlocks = m_blocks.size();
  BlockStructuredField_buildGridGPU<<<div_up(numBlocks, numThreads), numThreads>>>(
    vaccel, m_blocks.devicePtr(), numBlocks);
#else
  box3f worldBounds = {bounds().min,bounds().max};
  worldBounds.min = vfield.pointToVoxelSpace(worldBounds.min);
  worldBounds.max = vfield.pointToVoxelSpace(worldBounds.max);
  int3 dims{
    div_up(int(worldBounds.max.x-worldBounds.min.x),8),
    div_up(int(worldBounds.max.y-worldBounds.min.y),8),
    div_up(int(worldBounds.max.z-worldBounds.min.z),8)
  };
  m_gridAccel.init(dims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numBlocks = m_blocks.size();
  parallel::for_each(deviceState()->threadPool, 0, numBlocks,
    [&](size_t blockID) {
      const auto &block = m_blocks[blockID];
      int cellSize = block.cellSize();
      for (int z=0; z<block.numCells().z; ++z) {
        for (int y=0; y<block.numCells().y; ++y) {
          for (int x=0; x<block.numCells().x; ++x) {
            vec3i cellID(x,y,z);
            vec3i cell_lower = (block.bounds.min+cellID)*cellSize;
            vec3i cell_upper = (block.bounds.min+cellID+vec3i(1))*cellSize;
            aabb cellBounds(vec3f(cell_lower)+m_params.gridOrigin-vec3f(cellSize*0.5f),
                            vec3f(cell_upper)+m_params.gridOrigin+vec3f(cellSize*0.5f)); // +/- filterDomain
            float scalar = block.getScalar(x,y,z);

            const vec3i loMC = projectOnGrid(cellBounds.min,dims,worldBounds);
            const vec3i upMC = projectOnGrid(cellBounds.max,dims,worldBounds);

            for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
              for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
                for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
                  const vec3i mcID(mcx,mcy,mcz);
                  updateMC(mcID,vaccel.dims,scalar,vaccel.valueRanges);
                  updateMCStepSize(mcID,vaccel.dims,cellSize,vaccel.stepSizes);
                }
              }
            }
          }
        }
      }
  });
#endif
}

} // namespace visionaray
