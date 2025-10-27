// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "array/Array1D.h"
#include "array/ObjectArray.h"
#include "SpatialField.h"
#include "UElemGrid.h"

namespace visionaray {

struct UnstructuredField : public SpatialField
{
  UnstructuredField(VisionarayGlobalState *d);

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  aabb bounds() const override;

  void buildGrid() override;

 private:

  // first order unstructured elements
  HostDeviceArray<dco::UElem> m_elements;
  HostDeviceArray<float4> m_vertices;
  HostDeviceArray<uint64_t> m_indices;

  // vertex-centric voxel grids ("stitcher gridlets")
  HostDeviceArray<dco::UElemGrid> m_grids;
  HostDeviceArray<float> m_gridScalars;

  // sampling accels
#ifdef WITH_CUDA
  cuda_index_bvh<dco::UElem> m_elementBVH;
  cuda_index_bvh<dco::UElemGrid> m_gridBVH;
#else
  bvh4<dco::UElem> m_elementBVH;
  bvh4<dco::UElemGrid> m_gridBVH;
#endif

  // shell accel
#ifdef WITH_CUDA
  // must be an *index* BVH so we can access the
  // triangles based on their prim_id in the shader:
  cuda_index_bvh<basic_triangle<3,float>> m_shellBVH;
#else
  // must be an *index* BVH so we can access the
  // triangles based on their prim_id in the shader:
  index_bvh4<basic_triangle<3,float>> m_shellBVH;
#endif
  // for marching
  HostDeviceArray<uint64_t> m_faceNeighbors;

  struct Parameters
  {
    helium::IntrusivePtr<Array1D> vertexPosition;
    helium::IntrusivePtr<Array1D> vertexData;
    helium::IntrusivePtr<Array1D> index;
    helium::IntrusivePtr<Array1D> cellIndex;
    helium::IntrusivePtr<Array1D> cellType;
    helium::IntrusivePtr<Array1D> cellData;
    // "stitcher" extensions
    helium::IntrusivePtr<ObjectArray> gridData;
    helium::IntrusivePtr<Array1D> gridDomains;
  } m_params;
  anari::DataType m_type{ANARI_UNKNOWN};
};

} // namespace visionaray
