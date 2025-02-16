
#pragma onde

// visionaray
#include "visionaray/texture/texture.h"
// ours
#include "Volume.h"
#include "spatial_field/SpatialField.h"

namespace visionaray {

struct Blackbody : public Volume
{
  Blackbody(VisionarayGlobalState *d);
  ~Blackbody() override;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;

  bool isValid() const override;

  aabb bounds() const override;

 private:
  // Data //

  helium::ChangeObserverPtr<SpatialField> m_field;

  aabb m_bounds;

  box1 m_temperatureRange{1500.f, 6000.f};

#ifdef WITH_CUDA
  cuda_texture<float3, 1> colorTexture;
#elif defined(WITH_HIP)
  hip_texture<float3, 1> colorTexture;
#else
  texture<float3, 1> colorTexture;
#endif
};

} // namespace visionaray
