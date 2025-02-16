
// visionaray
#include <visionaray/detail/color_conversion.h>
// ours
#include "Blackbody.h"

namespace visionaray {

Blackbody::Blackbody(VisionarayGlobalState *d)
  : Volume(d)
  , m_field(this)
{
  vvol.type = dco::Volume::Blackbody;
}

Blackbody::~Blackbody()
{}

void Blackbody::commitParameters()
{
  Volume::commitParameters();

  m_field = getParamObject<SpatialField>("value");
  m_temperatureRange = getParam<box1>("temperatureRange", box1(1500.f, 6000.f));
}

void Blackbody::finalize()
{
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to blackbody volume");
    return;
  }

  if (!m_field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid spatial field provided to blackbody volume");
    return;
  }

  if (m_temperatureRange.min >= m_temperatureRange.max) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid temperature range for blackbody volume");
    return;
  }

  m_bounds = m_field->bounds();

  vvol.bounds = m_bounds;
  vvol.field = m_field->visionaraySpatialField();
  memset(&vvol.field.gridAccel,0,sizeof(vvol.field.gridAccel)); // NO grid accel!
  vvol.unitDistance = 1.f;//m_unitDistance;

  std::vector<float3> colors;
  for (float f = m_temperatureRange.min; f <= m_temperatureRange.max; f += 10.f) {
    blackbody spd(f);
    colors.push_back(spd_to_rgb(spd, 400.f, 700.f, 1.f, false));
  }

#if defined(WITH_CUDA) || defined(WITH_HIP)
  texture<float4, 1> tex(colors.size());
#else
  colorTexture = texture<float3, 1>(colors.size());
#endif
  auto &tex = colorTexture;
  tex.reset(colors.data());
  tex.set_filter_mode(Linear);
  tex.set_address_mode(Clamp);
#ifdef WITH_CUDA
  colorTexture.reset(tex);
  vvol.asBlackbody.sampler = cuda_texture_ref<float3, 1>(colorTexture);
#elif defined(WITH_HIP)
  colorTexture = hip_texture<float3, 1>(tex);
  vvol.asBlackbody.sampler = hip_texture_ref<float3, 1>(colorTexture);
#else
  vvol.asBlackbody.sampler = texture_ref<float3, 1>(colorTexture);
#endif

  // Trigger a BVH rebuild:
  deviceState()->objectUpdates.lastBLSReconstructSceneRequest = helium::newTimeStamp();

  dispatch();
}

void Blackbody::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLSCommitSceneRequest =
      helium::newTimeStamp();
}

bool Blackbody::isValid() const
{
  return m_field;
}

aabb Blackbody::bounds() const
{
  return m_bounds;
}

} // namespace visionaray
