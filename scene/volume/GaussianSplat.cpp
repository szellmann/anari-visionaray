// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "GaussianSplat.h"

namespace visionaray {

GaussianSplat::GaussianSplat(VisionarayGlobalState *d)
  : Volume(d)
  , m_splatBVH(d)
  , m_vertexPosition(this)
  , m_vertexScale(this)
  , m_vertexRotation(this)
  , m_vertexOpacity(this)
  , m_f_dc(this)
{
  vvol.type = dco::Volume::GaussianSplat;
}

void GaussianSplat::commitParameters()
{
  Volume::commitParameters();
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexScale = getParamObject<Array1D>("vertex.scale");
  m_vertexRotation = getParamObject<Array1D>("vertex.rotation");
  m_vertexOpacity = getParamObject<Array1D>("vertex.opacity");
  m_f_dc = getParamObject<Array1D>("vertex.f_dc");
}

void GaussianSplat::finalize()
{
  Volume::finalize();

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  const auto numSplats = m_vertexPosition->size();

  m_gaussians.resize(numSplats);

  const auto *vertices = m_vertexPosition->beginAs<float3>();
  const auto *scales = m_vertexScale->beginAs<float3>();
  const auto *rots = (float4 *)m_vertexRotation->data();
  const auto *opacities = m_vertexOpacity->beginAs<float>();

  m_bounds.invalidate();

  for (size_t i=0; i<numSplats; ++i) {
    const auto &v = vertices[i];
    m_gaussians[i].prim_id = uint32_t(i);
    m_gaussians[i].geom_id = -1;
    m_gaussians[i].pos = v * float3(1,-1,-1);
    m_gaussians[i].scale = scales[i];
    m_gaussians[i].rot = {rots[i].w, rots[i].x, -rots[i].y, -rots[i].z};
    m_gaussians[i].alpha = opacities[i];

    auto bbox = get_bounds(m_gaussians[i]);
    m_bounds.insert(bbox);
  }

  vvol.asGaussianSplat.primitives.data = m_gaussians.devicePtr();
  vvol.asGaussianSplat.primitives.len = m_gaussians.size();

  std::vector<float3> rgb;
  if (m_f_dc) {
    for (auto it = m_f_dc->beginAs<float3>(); it != m_f_dc->endAs<float3>(); ++it) {
      constexpr float SH_C0 = 0.28209479177387814f;
      float3 f_dc = *it;
      float3 color;
      color.x = std::clamp(f_dc.x * SH_C0 + 0.5f, 0.f, 1.f);
      color.y = std::clamp(f_dc.y * SH_C0 + 0.5f, 0.f, 1.f);
      color.z = std::clamp(f_dc.z * SH_C0 + 0.5f, 0.f, 1.f);
      color.x = powf(color.x, 2.2f);
      color.y = powf(color.y, 2.2f);
      color.z = powf(color.z, 2.2f);
      rgb.push_back(color);
    }

    m_colors.resize(rgb.size());
    m_colors.reset(rgb.data());

    vvol.asGaussianSplat.colors.data = m_colors.devicePtr();
    vvol.asGaussianSplat.colors.len = rgb.size();
    vvol.asGaussianSplat.colors.typeInfo = getInfo(ANARI_FLOAT32_VEC3);
  }

  m_splatBVH.update((const dco::Gaussian *)vvol.asGaussianSplat.primitives.data,
                    vvol.asGaussianSplat.primitives.len,
                    BVH_FLAG_ENABLE_SPATIAL_SPLITS | BVH_FLAG_NO_STREAM_SYNCHRONIZE);

  vvol.bounds = m_bounds;
  vvol.unitDistance = 1.f; // ignored (TODO?)
#if defined(WITH_CUDA) || defined(WITH_HIP)
  vvol.asGaussianSplat.splatBVH = m_splatBVH.deviceCWBVH();
#else
  vvol.asGaussianSplat.splatBVH = m_splatBVH.deviceBVH4();
#endif

  m_BVH.update(&vvol, 1, 0); // no spatial splits for volumes/aabbs

  vBLS.type = dco::BLS::Volume;
#if defined(WITH_CUDA) || defined(WITH_HIP)
  vBLS.asVolume = m_BVH.deviceBVH2();
#else
  vBLS.asVolume = m_BVH.deviceBVH4();
#endif

  // Trigger a BVH rebuild:
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest = helium::newTimeStamp();

  //dispatch();
  deviceState()->dcos.volumes.update(vvol.volID, vvol);
}

bool GaussianSplat::isValid() const
{
  return m_bounds.valid();
}

aabb GaussianSplat::bounds() const
{
  return m_bounds;
}

} // namespace visionaray
