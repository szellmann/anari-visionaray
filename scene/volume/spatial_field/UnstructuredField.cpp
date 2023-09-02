
#include "UnstructuredField.h"

namespace visionaray {

UnstructuredField::UnstructuredField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::Unstructured;
}

void UnstructuredField::commit()
{
  m_params.vertexPosition = getParamObject<Array1D>("vertex.position");
  m_params.vertexData = getParamObject<Array1D>("vertex.data");
  m_params.index = getParamObject<Array1D>("index");
  m_params.cellIndex = getParamObject<Array1D>("cell.index");

  if (!m_params.vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on unstructured spatial field");
    return;
  }

  if (!m_params.vertexData) { // currently vertex data only!
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.data' on unstructured spatial field");
    return;
  }

  if (!m_params.index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellIndex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'cell.index' on unstructured spatial field");
    return;
  }

  // TODO: check data type/index type validity!
  // cf. stagingBuffer in SR field?

  // Calculate bounds //

  size_t numVerts = m_params.vertexPosition->size();
  size_t numCells = m_params.cellIndex->size();

  m_vertices.resize(numVerts);
  m_elements.resize(numCells);

  auto *vertexPosition = m_params.vertexPosition->beginAs<vec3>();
  auto *vertexData = m_params.vertexData->beginAs<float>();
  auto *index = m_params.index->beginAs<uint64_t>();
  auto *cellIndex = m_params.cellIndex->beginAs<uint64_t>();

  size_t numIndices = m_params.index->endAs<uint64_t>()-index;

  for (size_t i=0; i<m_vertices.size(); ++i) {
    m_vertices[i] = float4(vertexPosition[i],vertexData[i]);
  }

  for (size_t cellID=0; cellID<m_elements.size(); ++cellID) {
    uint64_t firstIndex = cellIndex[cellID];
    uint64_t lastIndex = cellID < numCells-1 ? cellIndex[cellID+1] : numIndices;

    m_elements[cellID].begin = firstIndex;
    m_elements[cellID].end = lastIndex;
    m_elements[cellID].elemID = cellID;
    m_elements[cellID].vertexBuffer = m_vertices.data();
    m_elements[cellID].indexBuffer = index;
  }

  binned_sah_builder builder;
  builder.enable_spatial_splits(false);

  m_samplingBVH = builder.build(
    index_bvh<dco::UElem>{}, m_elements.data(), m_elements.size());

  vfield.asUnstructured.samplingBVH = m_samplingBVH.ref();

  setStepSize(length(bounds().max-bounds().min)/50.f);

  dispatch();
}

bool UnstructuredField::isValid() const
{
  return true;
}

aabb UnstructuredField::bounds() const
{
  return m_samplingBVH.node(0).get_bounds();
}

} // visionaray
