// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "StructuredRegularField.h"
// std
#include <limits>

namespace visionaray {

StructuredRegularField::StructuredRegularField(VisionarayGlobalState *d)
    : SpatialField(d)
{
  vfield.type = dco::SpatialField::StructuredRegular;
}

void StructuredRegularField::commitParameters()
{
  m_dataArray = getParamObject<Array3D>("data");
  m_origin = getParam<float3>("origin", float3(0.f));
  m_spacing = getParam<float3>("spacing", float3(1.f));
  m_filter = getParamString("filter", "linear");
}

void StructuredRegularField::finalize()
{
  if (!m_dataArray) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on 'structuredRegular' field");
    return;
  }

//m_data = m_dataArray->data();
  m_type = m_dataArray->elementType();
  m_dims = uint3(
      m_dataArray->size().x, m_dataArray->size().y, m_dataArray->size().z);

  mat3 S = mat3::scaling(1.f/bounds().size());
  vec3 T = -bounds().min;
  vfield.voxelSpaceTransform = mat4x3(S,T);
  setCellSize(min_element(m_spacing));

#if defined(WITH_CUDA) || defined(WITH_HIP)
  texture<float, 3> tex(m_dims.x, m_dims.y, m_dims.z);
#else
  m_dataTexture = texture<float, 3>(m_dims.x, m_dims.y, m_dims.z);
  auto &tex = m_dataTexture;
#endif
  if (m_type == ANARI_UFIXED8) {
    std::vector<float> data(m_dims.x * size_t(m_dims.y) * m_dims.z);
    auto data8 = (uint8_t *)m_dataArray->data();
    for (size_t i=0; i<data.size(); ++i) {
      data[i] = data8[i] / 255.f;
    }
    tex.reset(data.data());
  } else if (m_type == ANARI_UFIXED16) {
    std::vector<float> data(m_dims.x * size_t(m_dims.y) * m_dims.z);
    auto data16 = (uint16_t *)m_dataArray->data();
    for (size_t i=0; i<data.size(); ++i) {
      data[i] = data16[i] / 65535.f;
    }
    tex.reset(data.data());
  } else if (m_type == ANARI_FLOAT32)
    tex.reset((float *)m_dataArray->data());

  tex.set_filter_mode(m_filter == "nearest" ? Nearest : Linear);
  tex.set_address_mode(Clamp);

#ifdef WITH_CUDA
  m_dataTexture = cuda_texture<float, 3>(tex);
  vfield.asStructuredRegular.sampler = cuda_texture_ref<float, 3>(m_dataTexture);
#elif defined(WITH_HIP)
  m_dataTexture = hip_texture<float, 3>(tex);
  vfield.asStructuredRegular.sampler = hip_texture_ref<float, 3>(m_dataTexture);
#else
  vfield.asStructuredRegular.sampler = texture_ref<float, 3>(m_dataTexture);
#endif

  buildGrid();

  vfield.gridAccel = m_gridAccel.visionarayAccel();

  dispatch();
}

bool StructuredRegularField::isValid() const
{
  return m_dataArray && bounds().valid();
}

aabb StructuredRegularField::bounds() const
{
  return aabb(m_origin, m_origin + ((float3(m_dims) - 1.f) * m_spacing));
}

void StructuredRegularField::buildGrid()
{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return;
#endif
  int3 gridDims{16, 16, 16};
  box3f worldBounds = {bounds().min,bounds().max};
  // grid bounds are in "voxel space" (which for structured
  // volumes is just the [0:1] texture coordinate system:
  box3f gridBounds = worldBounds;
  gridBounds.min = vfield.pointToVoxelSpace(gridBounds.min);
  gridBounds.max = vfield.pointToVoxelSpace(gridBounds.max);
  m_gridAccel.init(gridDims, gridBounds);

  dco::GridAccel &vaccel = m_gridAccel.visionarayAccel();

  for (unsigned z=0; z<m_dims.z; ++z) {
    for (unsigned y=0; y<m_dims.y; ++y) {
      for (unsigned x=0; x<m_dims.x; ++x) {
        uint3 xyz{x,y,z};
        float3 P = m_origin + float3{xyz} * m_spacing;
        float3 texCoord = vfield.pointToVoxelSpace(P);
        float value = tex3D(vfield.asStructuredRegular.sampler, texCoord);
        box3f cellBounds{
          m_origin+float3{xyz}*m_spacing,
          m_origin+float3{xyz+uint3{1}}*m_spacing
        };

        const vec3i loMC = projectOnGrid(cellBounds.min,gridDims,worldBounds);
        const vec3i upMC = projectOnGrid(cellBounds.max,gridDims,worldBounds);

        for (int mcz=loMC.z; mcz<=upMC.z; ++mcz) {
          for (int mcy=loMC.y; mcy<=upMC.y; ++mcy) {
            for (int mcx=loMC.x; mcx<=upMC.x; ++mcx) {
              const vec3i mcID(mcx,mcy,mcz);
              updateMC(mcID,gridDims,value,vaccel.valueRanges);
              updateMCStepSize(
                  mcID,gridDims,min_element(m_spacing / 2.f),vaccel.stepSizes);
            }
          }
        }
      }
    }
  }
}

} // namespace visionaray
