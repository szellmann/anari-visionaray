// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "BlockStructuredField.h"
#include "for_each.h"
#include "array/Array3D.h"

namespace visionaray {

BlockStructuredField::BlockStructuredField(VisionarayGlobalState *d)
    : SpatialField(d), m_params(this)
{
  vfield.type = dco::SpatialField::BlockStructured;
}

void BlockStructuredField::commitParameters()
{
  m_params.blockOrigins = getParamObject<helium::Array1D>("block.origin");
  m_params.blockDims = getParamObject<helium::Array1D>("block.dimensions");
  m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
  m_params.data = getParamObject<helium::Array1D>("data");
  m_params.gridOrigin = getParam<float3>("origin", float3(0.f));
  m_params.gridSpacing = getParam<float3>("spacing", float3(1.f));
}

void BlockStructuredField::finalize()
{
  if (!m_params.blockOrigins) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.origin' on amr spatial field");
    return;
  }

  if (!m_params.blockDims) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.dimensions' on amr spatial field");
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

  size_t numBlocks = m_params.blockOrigins->totalSize();
  auto *blockOrigins = m_params.blockOrigins->beginAs<int3>();
  auto *blockDims = m_params.blockDims->beginAs<int3>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *data = m_params.data->beginAs<float>();

  m_scalars.clear();
  m_blocks.resize(numBlocks);

  std::vector<aabb> levelBounds;

  int maxLevel = -1;
  for (size_t i=0; i<numBlocks; ++i) {
    maxLevel = std::max(maxLevel,blockLevels[i]);
  }

  for (size_t i=0; i<numBlocks; ++i) {
    int3 blockSize = blockDims[i];

    m_blocks[i].bounds = aabbi(blockOrigins[i], blockOrigins[i] + blockDims[i]);
    m_blocks[i].level = blockLevels[i];
    m_blocks[i].scalarOffset = m_scalars.size();
    m_blocks[i].valueRange = box1f(FLT_MAX,-FLT_MAX);

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

  auto samplingBVH2 = builder.build(bvh<dco::Block>{}, m_blocks.data(), m_blocks.size());

#ifdef WITH_CUDA
  m_samplingBVH = cuda_bvh<dco::Block>(samplingBVH2);
#else
  bvh_collapser collapser;
  collapser.collapse(samplingBVH2, m_samplingBVH, deviceState()->syncContext->threadPool);
#endif

  vfield.asBlockStructured.samplingBVH = m_samplingBVH.ref();

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
  float cellSize = block.cellSize();

  for (int z=0; z<block.numCells().z; ++z) {
    for (int y=0; y<block.numCells().y; ++y) {
      for (int x=0; x<block.numCells().x; ++x) {
        vec3i cellID(x,y,z);
        vec3f cell_lower = vec3f(block.bounds.min+cellID)*cellSize;
        vec3f cell_upper = vec3f(block.bounds.min+cellID+vec3i(1))*cellSize;
        box3f cellBounds(cell_lower-vec3f(cellSize*0.5f),
                         cell_upper+vec3f(cellSize*0.5f)); // +/- filterDomain
        float scalar = block.getScalar(x,y,z);

        rasterizeBox(vaccel,cellBounds,box1f(scalar),cellSize);
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
  box3f gridBounds = worldBounds;
  m_gridAccel.init(dims, worldBounds, gridBounds);

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
  box3f gridBounds = worldBounds;
  m_gridAccel.init(dims, worldBounds, gridBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numBlocks = m_blocks.size();
  parallel::for_each(deviceState()->syncContext->threadPool, 0, numBlocks,
    [&](size_t blockID) {
      const auto &block = m_blocks[blockID];
      float cellSize = block.cellSize();
      for (int z=0; z<block.numCells().z; ++z) {
        for (int y=0; y<block.numCells().y; ++y) {
          for (int x=0; x<block.numCells().x; ++x) {
            vec3i cellID(x,y,z);
            vec3f cell_lower = vec3f(block.bounds.min+cellID)*cellSize;
            vec3f cell_upper = vec3f(block.bounds.min+cellID+vec3i(1))*cellSize;
            box3f cellBounds(cell_lower+m_params.gridOrigin-vec3f(cellSize*0.5f),
                             cell_upper+m_params.gridOrigin+vec3f(cellSize*0.5f)); // +/- filterDomain
            float scalar = block.getScalar(x,y,z);

            rasterizeBox(vaccel,cellBounds,box1f(scalar),cellSize);
          }
        }
      }
  });
#endif
}

} // namespace visionaray
