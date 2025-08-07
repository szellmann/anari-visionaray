
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
  m_elements.resize(numCells);

  auto *vertexPosition = m_params.vertexPosition->beginAs<vec3>();
  auto *vertexData = m_params.vertexData ? m_params.vertexData->beginAs<float>() : nullptr;
  auto *cellType = m_params.cellType->beginAs<uint8_t>();
  auto *cellData = m_params.cellData ? m_params.cellData->beginAs<float>() : nullptr;

  uint32_t *index = (uint32_t *)m_params.index->beginAs<uint32_t>();
  uint32_t *cellIndex = (uint32_t *)m_params.cellIndex->beginAs<uint32_t>();

  // try to guess how to interpret the cell type, as this is
  // not properly specified yet:

  auto indexCount = [this](uint8_t val) {
    // Banari: 0,1,2,3
    // VTK: 10,12,13,14
    if (val == 0 || val == 10) return 4ull; // TET
    if (val == 1 || val == 12) return 8ull; // HEX
    if (val == 2 || val == 13) return 6ull; // WEDGE
    if (val == 3 || val == 14) return 5ull; // PYR
    else {
      reportMessage(ANARI_SEVERITY_WARNING,
        "unknown value for 'cell.type' found, returning 0 for index count");
      return ~0ull;
    }
  };

#if 1
  for (size_t i=0; i<m_vertices.size(); ++i) {
    float value = (vertexData != nullptr) ? vertexData[i] : NAN;
    m_vertices[i] = float4(vertexPosition[i],value);
  }
#else
  float minDistance = FLT_MAX;
  for (size_t i=0; i<m_vertices.size(); ++i) {
    float value = (vertexData != nullptr) ? vertexData[i] : NAN;
    m_vertices[i] = float4(vertexPosition[i],value);
    float3 pos(vertexPosition[i].x,vertexPosition[i].y,vertexPosition[i].z);
    minDistance = std::min(minDistance,length(pos));
  }

  for (size_t i=0; i<m_vertices.size(); ++i) {
    float3 pos(vertexPosition[i].x,vertexPosition[i].y,vertexPosition[i].z);
    float3 norm = normalize(pos);
    pos -= norm*minDistance;
    pos *= 30.f;
    pos += norm*minDistance;
    m_vertices[i].x = pos.x;
    m_vertices[i].y = pos.y;
    m_vertices[i].z = pos.z;
  }
#endif

  for (size_t i=0; i<m_indices.size(); ++i) {
    m_indices[i] = uint64_t(index[i]);
  }

  uint64_t currentIndex=0;
  float minCellDiagonal=FLT_MAX;
  for (size_t cellID=0; cellID<m_elements.size(); ++cellID) {
    uint64_t firstIndex, lastIndex;

    if (cellIndex) {
      firstIndex = cellIndex[cellID];
      lastIndex = cellID < numCells-1 ? cellIndex[cellID+1] : numIndices;
    } else /*if (cellType) */ {
      auto ic = indexCount(cellType[cellID]);
      firstIndex = currentIndex;
      lastIndex = currentIndex + ic;
      currentIndex += ic;
    }

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
  }

  // voxel grid extensions for AMR "stitching"
  m_params.gridData = getParamObject<ObjectArray>("grid.data");
  m_params.gridDomains = getParamObject<Array1D>("grid.domains");

  if (m_params.gridData && m_params.gridDomains) {
    m_gridDims.clear();
    m_gridDomains.clear();
    m_gridScalarsOffsets.clear();
    m_gridScalars.clear();

    size_t numGrids = m_params.gridData->totalSize();
    auto *gridData = (Array3D **)m_params.gridData->handlesBegin();
    auto *gridDomains = m_params.gridDomains->beginAs<aabb>();

    for (size_t i=0; i<numGrids; ++i) {
      const Array3D *gd = *(gridData+i);

      // from anari's array3d we get the number of vertices, not cells!
      m_gridDims.push_back(int3(gd->size().x-1,gd->size().y-1,gd->size().z-1));
      m_gridDomains.push_back(*(gridDomains+i));
      m_gridScalarsOffsets.push_back(m_gridScalars.size());

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

    uint64_t firstGridID = m_elements.size();
    for (size_t i=0; i<numGrids; ++i) {
      dco::UElem elem;
      elem.begin = elem.end = 0; // denotes that this is a grid!
      elem.elemID = /*firstGridID +*/ i;
      elem.gridDimsBuffer = m_gridDims.devicePtr();
      elem.gridDomainsBuffer = m_gridDomains.devicePtr();
      elem.gridScalarsOffsetBuffer = m_gridScalarsOffsets.devicePtr();
      elem.gridScalarsBuffer = m_gridScalars.devicePtr();
      m_elements.push_back(elem);
    }
  }

  // connectivity for element marcher

  conn::Mesh connMesh(m_vertices.hostPtr(),
                      m_indices.hostPtr(),
                      m_elements.hostPtr(),
                      numCells);

  auto fn = computeFaceConnectivity(connMesh);
  // TODO:!!!!!
  m_faceNeighbors.resize(fn.size());
  m_faceNeighbors.reset(fn.data());

  auto shell = computeShell(connMesh,fn.data());

#if 0
  for (size_t i=0; i<shell.size(); ++i) {
    float3 v1 = shell[i].v1;
    float3 v2 = shell[i].e1 + v1;
    float3 v3 = shell[i].e2 + v1;

    std::cout << "v " << v1.x << ' ' << v1.y << ' ' << v1.z << '\n';
    std::cout << "v " << v2.x << ' ' << v2.y << ' ' << v2.z << '\n';
    std::cout << "v " << v3.x << ' ' << v3.y << ' ' << v3.z << '\n';
  }

  for (size_t i=0; i<shell.size(); ++i) {
    std::cout << "f " << (i*3)+1 << ' ' << (i*3)+2 << ' ' << (i*3)+3 << '\n';
  }
#endif

  // build shell BVH, init marcher:
  {
    binned_sah_builder builder;
    builder.enable_spatial_splits(false);

    auto shellBVH2 = builder.build(
      index_bvh<basic_triangle<3,float>>{}, shell.data(), shell.size());

    bvh_collapser collapser;
    collapser.collapse(shellBVH2, m_shellBVH, deviceState()->threadPool);

    vfield.asUnstructured.shellBVH = m_shellBVH.ref();
    vfield.asUnstructured.elems = m_elements.hostPtr(); // devicePtr() here but the data isn't on the device......
    vfield.asUnstructured.faceNeighbors = m_faceNeighbors.hostPtr();
  }

  // sampling BVH

#ifdef WITH_CUDA
  lbvh_builder builder; // the only GPU builder Visionaray has (for now..)
  m_samplingBVH = builder.build(
    cuda_index_bvh<dco::UElem>{}, m_elements.devicePtr(), m_elements.size());

  vfield.asUnstructured.samplingBVH = m_samplingBVH.ref();
#else
  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

  auto samplingBVH2 = builder.build(
    bvh<dco::UElem>{}, m_elements.data(), m_elements.size());

  bvh_collapser collapser;
  collapser.collapse(samplingBVH2, m_samplingBVH, deviceState()->threadPool);

  vfield.asUnstructured.samplingBVH = m_samplingBVH.ref();
#endif

  vfield.voxelSpaceTransform = mat4x3(mat3::identity(),vec3f(0.f));
  setCellSize(minCellDiagonal);

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

  dispatch();
}

bool UnstructuredField::isValid() const
{
  return m_samplingBVH.num_nodes();
}

aabb UnstructuredField::bounds() const
{
#ifdef WITH_CUDA
  if (isValid()) {
    bvh_node rootNode;
    CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                              m_samplingBVH.nodes().data(),
                              sizeof(rootNode),
                              cudaMemcpyDeviceToHost));
    return rootNode.get_bounds();
  }
#else
  if (isValid())
    return m_samplingBVH.node(0).get_bounds();
#endif
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

  uint64_t numVertices = elements[cellID].end-elements[cellID].begin;
  if (numVertices > 0) {
    for (uint64_t i=elements[cellID].begin; i<elements[cellID].end; ++i) {
      const vec4f V = vertices[elements[cellID].indexBuffer[i]];
      cellBounds.extend(V.xyz());
      valueRange.extend(V.w);
    }
  } else { // grid!
    assert(0 && "Not implemented yet!");
  }

  const vec3i loMC = projectOnGrid(cellBounds.min,vaccel.dims,vaccel.worldBounds);
  const vec3i upMC = projectOnGrid(cellBounds.max,vaccel.dims,vaccel.worldBounds);

  for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
    for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
      for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
        const vec3i mcID(mcx,mcy,mcz);
        updateMC(mcID,vaccel.dims,valueRange,vaccel.valueRanges);
        // TODO: this causes artifacts, should probably do this in
        // a macrocell neighborhood:
        //updateMCStepSize(
        //    mcID,vaccel.dims,length(cellBounds.max-cellBounds.min),vaccel.stepSizes);
        updateMCStepSize(mcID,vaccel.dims,cellSize,vaccel.stepSizes);
      }
    }
  }
}
#endif

void UnstructuredField::buildGrid()
{
#ifdef WITH_CUDA
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(dims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  size_t numThreads = 1024;
  size_t numElems = m_elements.size();
  UnstructuredField_buildGridGPU<<<div_up(numElems, numThreads), numThreads>>>(
    vaccel, m_vertices.devicePtr(), m_elements.devicePtr(), numElems, vfield.cellSize);
#else
  int3 dims{64, 64, 64};
  box3f worldBounds = {bounds().min,bounds().max};
  m_gridAccel.init(dims, worldBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  for (size_t cellID=0; cellID<m_elements.size(); ++cellID) {
    box3f cellBounds{vec3{FLT_MAX}, vec3{-FLT_MAX}};
    box1f valueRange{FLT_MAX, -FLT_MAX};

    uint64_t numVertices = m_elements[cellID].end-m_elements[cellID].begin;
    if (numVertices > 0) {
      for (uint64_t i=m_elements[cellID].begin; i<m_elements[cellID].end; ++i) {
        const vec4f V = m_vertices[m_elements[cellID].indexBuffer[i]];
        cellBounds.extend(V.xyz());
        valueRange.extend(V.w);
      }
    } else { // grid!
      uint64_t elemID = m_elements[cellID].elemID; // not unique! (grids are 0-based)
      cellBounds = box3f(vec3f(m_gridDomains[elemID].min.x,
                               m_gridDomains[elemID].min.y,
                               m_gridDomains[elemID].min.z),
                         vec3f(m_gridDomains[elemID].max.x,
                               m_gridDomains[elemID].max.y,
                               m_gridDomains[elemID].max.z));
  
      int3 dims = m_gridDims[elemID];

      uint64_t numScalars = (dims.x+1)*size_t(dims.y+1)*(dims.z+1);
      for (uint64_t i=0; i<numScalars; ++i) {
        float f = m_gridScalars[m_gridScalarsOffsets[elemID] + i];
        valueRange.extend(f);
      }
    }

    const vec3i loMC = projectOnGrid(cellBounds.min,vaccel.dims,vaccel.worldBounds);
    const vec3i upMC = projectOnGrid(cellBounds.max,vaccel.dims,vaccel.worldBounds);

    for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
      for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
        for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
          const vec3i mcID(mcx,mcy,mcz);
          updateMC(mcID,vaccel.dims,valueRange,vaccel.valueRanges);
          // TODO: this causes artifacts, should probably do this in
          // a macrocell neighborhood:
          //updateMCStepSize(
          //    mcID,vaccel.dims,length(cellBounds.max-cellBounds.min),vaccel.stepSizes);
          updateMCStepSize(
              mcID,vaccel.dims,vfield.cellSize,vaccel.stepSizes);
        }
      }
    }
  }
#endif
}

} // visionaray
