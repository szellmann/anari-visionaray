// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "array/Array3D.h"
#include "Connectivity.h"
#include "UnstructuredField.h"

namespace visionaray {

UnstructuredField::UnstructuredField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::Unstructured;
}

void UnstructuredField::commitParameters()
{
  m_params.vertexPosition = getParamObject<Array1D>("vertex.position");
  m_params.vertexData = getParamObject<Array1D>("vertex.data");
  m_params.index = getParamObject<Array1D>("index");
  m_params.cellIndex = getParamObject<Array1D>("cell.index");
  m_params.cellType = getParamObject<Array1D>("cell.type");
  m_params.cellData = getParamObject<Array1D>("cell.data");
  // voxel grid extensions for AMR "stitching"
  m_params.gridData = getParamObject<ObjectArray>("grid.data");
  m_params.gridDomains = getParamObject<Array1D>("grid.domains");
}

void UnstructuredField::finalize()
{
  if (!m_params.vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on unstructured spatial field");
    return;
  }

  if (!(m_params.vertexData || m_params.cellData)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.data' (or 'cellData') on unstructured spatial field");
    return;
  }

  if (!m_params.index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellIndex && !m_params.cellType) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter(s) 'cell.index' and 'cell.type' on unstructured spatial field");
    return;
  }

  if (m_params.cellIndex && !m_params.cellType) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no 'cell.type' specified on unstructured spatial field; trying to guess from 'cell.index'");
  }

  // TODO: check data type/index type validity!
  // cf. stagingBuffer in SR field?

  // Calculate bounds //

  size_t numVerts = m_params.vertexPosition->size();
  size_t numIndices = m_params.index->size();
  size_t numCells = m_params.cellIndex ? m_params.cellIndex->size()
                                       : m_params.cellType->size();

  m_vertices.resize(numVerts);
  m_indices.resize(numIndices);

  auto *vertexPosition = m_params.vertexPosition->beginAs<vec3>();
  auto *vertexData = m_params.vertexData ? m_params.vertexData->beginAs<float>() : nullptr;
  auto *cellType = m_params.cellType->beginAs<uint8_t>();
  auto *cellData = m_params.cellData ? m_params.cellData->beginAs<float>() : nullptr;

  uint32_t *index = (uint32_t *)m_params.index->beginAs<uint32_t>();
  uint32_t *cellIndex = (uint32_t *)m_params.cellIndex->beginAs<uint32_t>();

  // Not sure if these are used anymore:
  enum {
    BARNEY_TET_ = 0,
    BARNEY_HEX_ = 1,
    BARNEY_WEDGE_ = 2,
    BARNEY_PYR_ = 3,
  };

  enum {
    VTK_TET_ = 10,
    VTK_HEX_ = 12,
    VTK_WEDGE_ = 13,
    VTK_PYR_ = 14,
    VTK_BEZIER_HEX_ = 79,
  };

  auto mapType = [this](uint8_t val) {
    if (val == BARNEY_TET_ || val == VTK_TET_) return dco::UElem::Tet;
    if (val == BARNEY_HEX_ || val == VTK_HEX_) return dco::UElem::Hex;
    if (val == BARNEY_WEDGE_ || val == VTK_WEDGE_) return dco::UElem::Wedge;
    if (val == BARNEY_PYR_ || val == VTK_PYR_) return dco::UElem::Pyr;
    if (val == VTK_BEZIER_HEX_) return dco::UElem::BezierHex;
    return dco::UElem::Unknown;
  };

  for (size_t i=0; i<m_vertices.size(); ++i) {
    float value = (vertexData != nullptr) ? vertexData[i] : NAN;
    m_vertices[i] = float4(vertexPosition[i],value);
  }

  for (size_t i=0; i<m_indices.size(); ++i) {
    m_indices[i] = uint64_t(index[i]);
  }

  // build first order element list //

  m_elements.clear();

  uint64_t currentIndex=0;
  float minCellDiagonal=FLT_MAX;
  float avgCellDiagonal=0.f;
  for (size_t cellID=0; cellID<numCells; ++cellID) {
    uint64_t firstIndex, lastIndex;

    dco::UElem::Type elemType;

    if (cellType) {
      elemType = mapType(cellType[cellID]);
    }

    firstIndex = cellIndex[cellID];
    lastIndex = cellID < numCells-1 ? cellIndex[cellID+1] : numIndices;
    uint64_t ic = lastIndex-firstIndex;

    if (ic < 4) {
      reportMessage(ANARI_SEVERITY_WARNING, "Invalid element with (%i) indices", ic);
      continue;
    }

    if ((elemType == dco::UElem::Tet && ic != 4) ||
      (elemType == dco::UElem::Pyr && ic != 5) ||
      (elemType == dco::UElem::Wedge && ic != 6) ||
      (elemType == dco::UElem::Hex && ic != 8)) {
      reportMessage(ANARI_SEVERITY_WARNING,
        "'cell.type' (%i) and index count (%i) do not match", (int)elemType, ic);
      continue;
    }

    m_elements.emplace_back();
    m_elements[cellID].type = elemType;
    m_elements[cellID].begin = firstIndex;
    m_elements[cellID].end = lastIndex;
    m_elements[cellID].elemID = cellID;
    m_elements[cellID].vertexBuffer = m_vertices.devicePtr();
    m_elements[cellID].indexBuffer = m_indices.devicePtr();

    m_elements[cellID].cellValue = 0.f;
    if (cellData != nullptr) {
      m_elements[cellID].cellValue = cellData[cellID];
    }

    // compute cellBounds; the minimum size will determine the
    // step size used for (ISO) gradient shading:
    box3f cellBounds{vec3{FLT_MAX}, vec3{-FLT_MAX}};
    for (uint64_t i=firstIndex; i<lastIndex; ++i) {
      const vec4f V = m_vertices.hostPtr()[m_indices.hostPtr()[i]];
      cellBounds.extend(V.xyz());
    }
    minCellDiagonal = fminf(minCellDiagonal,length(cellBounds.max-cellBounds.min));
    avgCellDiagonal += length(cellBounds.max-cellBounds.min)/m_elements.size();
  }

  // build AMR gridlet list //

  m_grids.clear();

  size_t numGrids = m_params.gridData ? m_params.gridData->totalSize() : 0;
  auto *gridData = m_params.gridData ? (Array3D **)m_params.gridData->handlesBegin() : nullptr;
  auto *gridDomains = m_params.gridDomains ? m_params.gridDomains->beginAs<aabb>() : nullptr;

  for (size_t gridID=0; gridID<numGrids; ++gridID) {
    const Array3D *gd = *(gridData+gridID);

    m_grids.emplace_back();
    m_grids[gridID].gridID = gridID;
    m_grids[gridID].dims = int3(gd->size().x-1,gd->size().y-1,gd->size().z-1);
    m_grids[gridID].domain = *(gridDomains+gridID);
    m_grids[gridID].scalarsOffset = (uint64_t)m_gridScalars.size();

    // build scalar array
    for (unsigned z=0;z<gd->size().z;++z) {
      for (unsigned y=0;y<gd->size().y;++y) {
        for (unsigned x=0;x<gd->size().x;++x) {
          // TODO: can we actually iterate linearly here?!
          size_t idx = z*size_t(gd->size().x)*gd->size().y
                       + y*gd->size().x
                       + x;
          float f = gd->dataAs<float>()[idx];
          m_gridScalars.push_back(f);
        }
      }
    }
  }

  // second pass, set each grid's scalar buffer pointer
  for (size_t gridID=0; gridID<numGrids; ++gridID) {
    m_grids[gridID].scalarsBuffer = m_gridScalars.devicePtr();
  }

  // connectivity for element marcher

  conn::Mesh connMesh(m_vertices.hostPtr(),
                      m_indices.hostPtr(),
                      m_elements.hostPtr(),
                      m_elements.size());

  auto fn = computeFaceConnectivity(connMesh);
  // TODO:!!!!!
  m_faceNeighbors.resize(fn.size());
  m_faceNeighbors.reset(fn.data());

  auto shell = computeShell(connMesh,fn.data());

  // build shell BVH, init marcher:
  if (shell.size() > 0) {
    binned_sah_builder builder;
    builder.enable_spatial_splits(false);

    auto shellBVH2 = builder.build(
      index_bvh<basic_triangle<3,float>>{}, shell.data(), shell.size());

#if defined(WITH_CUDA)
    m_shellBVH = cuda_index_bvh<basic_triangle<3,float>>(shellBVH2);
#elif defined(WITH_HIP)
    m_shellBVH = hip_index_bvh<basic_triangle<3,float>>(shellBVH2);
#elif defined(WITH_SYCL)
    m_shellBVH = sycl_index_bvh<basic_triangle<3,float>>(shellBVH2);
#else
    bvh_collapser collapser;
    collapser.collapse(shellBVH2, m_shellBVH, deviceState()->threadPool);
#endif

    vfield.asUnstructured.shellBVH = m_shellBVH.ref();
    vfield.asUnstructured.elems = m_elements.devicePtr(); // devicePtr() here but the data isn't on the device......
    vfield.asUnstructured.faceNeighbors = m_faceNeighbors.devicePtr();
  }

  // sampling BVHs

#if defined(WITH_CUDA)
  lbvh_builder builder; // the only GPU builder Visionaray has (for now..)

  if (!m_elements.empty()) {
    m_elementBVH = builder.build(
      cuda_index_bvh<dco::UElem>{}, m_elements.devicePtr(), m_elements.size());
  }

  if (!m_grids.empty()) {
    m_gridBVH = builder.build(
      cuda_index_bvh<dco::UElemGrid>{}, m_grids.devicePtr(), m_grids.size());
  }
#elif defined(WITH_HIP)
  lbvh_builder builder; // the only GPU builder Visionaray has (for now..)

  if (!m_elements.empty()) {
    m_elementBVH = builder.build(
      hip_index_bvh<dco::UElem>{}, m_elements.devicePtr(), m_elements.size());
  }

  if (!m_grids.empty()) {
    m_gridBVH = builder.build(
      hip_index_bvh<dco::UElemGrid>{}, m_grids.devicePtr(), m_grids.size());
  }
#elif defined(WITH_SYCL)
  lbvh_builder builder; // the only GPU builder Visionaray has (for now..)

  if (!m_elements.empty()) {
    m_elementBVH = builder.build(
      sycl_index_bvh<dco::UElem>{}, m_elements.devicePtr(), m_elements.size());
  }

  if (!m_grids.empty()) {
    m_gridBVH = builder.build(
      sycl_index_bvh<dco::UElemGrid>{}, m_grids.devicePtr(), m_grids.size());
  }
#else
  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

  bvh_collapser collapser;

  if (!m_elements.empty()) {
    auto elemBVH2 = builder.build(
      bvh<dco::UElem>{}, m_elements.data(), m_elements.size());
    collapser.collapse(elemBVH2, m_elementBVH, deviceState()->threadPool);
  }

  if (!m_grids.empty()) {
    auto gridBVH2 = builder.build(
      bvh<dco::UElemGrid>{}, m_grids.data(), m_grids.size());
    collapser.collapse(gridBVH2, m_gridBVH, deviceState()->threadPool);
  }
#endif

  vfield.asUnstructured.elemBVH = m_elementBVH.ref();
  vfield.asUnstructured.gridBVH = m_gridBVH.ref();

  vfield.voxelSpaceTransform = mat4x3(mat3::identity(),vec3f(0.f));
  setCellSize(avgCellDiagonal);

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

  dispatch();
}

bool UnstructuredField::isValid() const
{
  return m_elementBVH.num_nodes() || m_gridBVH.num_nodes();
}

aabb UnstructuredField::bounds() const
{
  if (isValid()) {
    aabb bounds;
    bounds.invalidate();

#if defined(WITH_CUDA)
    if (m_elementBVH.num_nodes()) {
      bvh_node rootNode;
      CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                                m_elementBVH.nodes().data(),
                                sizeof(rootNode),
                                cudaMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }

    if (m_gridBVH.num_nodes()) {
      bvh_node rootNode;
      CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                                m_gridBVH.nodes().data(),
                                sizeof(rootNode),
                                cudaMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }
#elif defined(WITH_HIP)
    if (m_elementBVH.num_nodes()) {
      bvh_node rootNode;
      HIP_SAFE_CALL(hipMemcpy(&rootNode,
                              m_elementBVH.nodes().data(),
                              sizeof(rootNode),
                              hipMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }

    if (m_gridBVH.num_nodes()) {
      bvh_node rootNode;
      HIP_SAFE_CALL(hipMemcpy(&rootNode,
                              m_gridBVH.nodes().data(),
                              sizeof(rootNode),
                              hipMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }
#elif defined(WITH_SYCL)
    if (m_elementBVH.num_nodes()) {
      bvh_node rootNode;
      //CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
      //                          m_elementBVH.nodes().data(),
      //                          sizeof(rootNode),
      //                          cudaMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }

    if (m_gridBVH.num_nodes()) {
      bvh_node rootNode;
      //CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
      //                          m_gridBVH.nodes().data(),
      //                          sizeof(rootNode),
      //                          cudaMemcpyDeviceToHost));
      bounds.insert(rootNode.get_bounds());
    }
#else
    if (m_elementBVH.num_nodes())
      bounds.insert(m_elementBVH.node(0).get_bounds());

    if (m_gridBVH.num_nodes())
      bounds.insert(m_gridBVH.node(0).get_bounds());

#endif

    return bounds;
  } // isValid

  return {};
}

#ifdef WITH_CUDA
__global__ void UnstructuredField_buildGridGPU(dco::GridAccel    vaccel,
                                               const vec4f      *vertices,
                                               const dco::UElem *elements,
                                               size_t            numElems,
                                               float             cellSize)
{
  size_t cellID = blockIdx.x * size_t(blockDim.x) + threadIdx.x;

  if (cellID >= numElems)
    return;

// TODO: deduplicate, refactor into function
  box3f cellBounds{vec3{FLT_MAX}, vec3{-FLT_MAX}};
  box1f valueRange{FLT_MAX, -FLT_MAX};

  for (uint64_t i=elements[cellID].begin; i<elements[cellID].end; ++i) {
    const vec4f V = vertices[elements[cellID].indexBuffer[i]];
    cellBounds.extend(V.xyz());
    valueRange.extend(V.w);
  }

  rasterizeBox(vaccel,cellBounds,valueRange,cellSize);
}
#endif

void UnstructuredField::buildGrid()
{
#ifdef WITH_CUDA
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  box3f gridBounds = worldBounds;
  m_gridAccel.init(dims, worldBounds, gridBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numThreads = 1024;
  size_t numElems = m_elements.size();
  UnstructuredField_buildGridGPU<<<div_up(numElems, numThreads), numThreads>>>(
    vaccel, m_vertices.devicePtr(), m_elements.devicePtr(), numElems, vfield.cellSize);
  // TODO: stitcher gridlets
#else
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  box3f gridBounds = worldBounds;
  m_gridAccel.init(dims, worldBounds, gridBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  for (size_t cellID=0; cellID<m_elements.size(); ++cellID) {
    box3f cellBounds{vec3{FLT_MAX}, vec3{-FLT_MAX}};
    box1f valueRange{FLT_MAX, -FLT_MAX};

    for (uint64_t i=m_elements[cellID].begin; i<m_elements[cellID].end; ++i) {
      const vec4f V = m_vertices[m_elements[cellID].indexBuffer[i]];
      cellBounds.extend(V.xyz());
      valueRange.extend(V.w);
    }

    rasterizeBox(vaccel,cellBounds,valueRange,vfield.cellSize);
  }

  for (size_t gridID=0; gridID<m_grids.size(); ++gridID) {
    box3f cellBounds(m_grids[gridID].domain.min, m_grids[gridID].domain.max);
    int3 dims = m_grids[gridID].dims;

    box1f valueRange{FLT_MAX, -FLT_MAX};

    uint64_t numScalars = (dims.x+1)*size_t(dims.y+1)*(dims.z+1);
    for (uint64_t i=0; i<numScalars; ++i) {
      float f = m_gridScalars[m_grids[gridID].scalarsOffset + i];
      valueRange.extend(f);
    }

    rasterizeBox(vaccel,cellBounds,valueRange,vfield.cellSize);
  }
#endif
}

} // visionaray
