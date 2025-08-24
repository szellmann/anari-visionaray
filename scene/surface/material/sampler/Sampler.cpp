// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// subtypes
#include "Image1D.h"
#include "Image2D.h"
#include "Image3D.h"
#include "PrimitiveSampler.h"
#include "TransformSampler.h"
#include "VolumeSampler.h"

namespace visionaray {

Sampler::Sampler(VisionarayGlobalState *s) : Object(ANARI_SAMPLER, s)
{
  vsampler = dco::createSampler();
  vsampler.samplerID = deviceState()->dcos.samplers.alloc(vsampler);
}

Sampler::~Sampler()
{
  deviceState()->dcos.samplers.free(vsampler.samplerID);
}

Sampler *Sampler::createInstance(std::string_view subtype, VisionarayGlobalState *s)
{
  if (subtype == "image1D")
    return new Image1D(s);
  else if (subtype == "image2D")
    return new Image2D(s);
  else if (subtype == "image3D")
    return new Image3D(s);
  else if (subtype == "transform")
    return new TransformSampler(s);
  else if (subtype == "primitive")
    return new PrimitiveSampler(s);
  else if (subtype == "volume")
    return new VolumeSampler(s);
  else
    return (Sampler *)new UnknownObject(ANARI_SAMPLER, s);
}

bool Sampler::isValid() const
{
  return vsampler.isValid();
}

dco::Sampler Sampler::visionaraySampler() const
{
  return vsampler;
}

void Sampler::dispatch()
{
  deviceState()->dcos.samplers.update(vsampler.samplerID, vsampler);

  // Upload/set accessible pointers
  deviceState()->onDevice.samplers = deviceState()->dcos.samplers.devicePtr();
}

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_DEFINITION(visionaray::Sampler *);
