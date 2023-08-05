
#pragma once

// visionaray
#include "visionaray/aligned_vector.h"
#include "visionaray/bvh.h"
// ours
#include "surface/geometry/Geometry.h"
#include "surface/material/Material.h"

namespace visionaray {

struct VisionarayScene
{
  // Geometries //
  aligned_vector<VisionarayGeometry> m_geometries;
  aligned_vector<index_bvh<basic_triangle<3,float>>> m_triangleBLSs;
  index_bvh<index_bvh<basic_triangle<3,float>>> m_triangleTLS;

  // Surface properties //
  aligned_vector<VisionarayMaterial> m_materials;

  void commit();
  void release();
  void attachGeometry(VisionarayGeometry geom, unsigned geomID);
};

} // namespace visionaray
