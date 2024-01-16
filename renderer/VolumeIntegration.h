#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline float4 postClassify(ScreenSample &ss, dco::TransferFunction tf, float v) {
  if (tf.type == dco::TransferFunction::_1D) {
    box1 valueRange = tf.as1D.valueRange;
    v = (v - valueRange.min) / (valueRange.max - valueRange.min);
    float4 clr = tex1D(tf.as1D.sampler, v);
    // if (ss.debug()) {
    //   printf("v: %f, clr: (%f,%f,%f)\n",v,clr.x,clr.y,clr.z);
    // }
    return clr;
  }

  return {};
}

VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            Ray ray,
                            const dco::Volume &vol,
                            VisionarayGlobalState::DeviceObjectRegistry onDevice,
                            float3 &color,
                            float &alpha) {
  float dt = onDevice.spatialFields[vol.asTransferFunction1D.fieldID].baseDT;
  auto boxHit = intersect(ray, vol.bounds);
  // if (ss.debug()) {
  //   printf("boxHit: %f,%f\n",boxHit.tnear,boxHit.tfar);
  //   print(ray);
  //   print(vol.bounds);
  // }
  float t=boxHit.tnear;
  for (;t<boxHit.tfar&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(onDevice.spatialFields[vol.asTransferFunction1D.fieldID],P,v)) {
      float4 sample
          = postClassify(ss,onDevice.transferFunctions[vol.asTransferFunction1D.tfID],v);
      color += dt * (1.f-alpha) * sample.w * sample.xyz();
      alpha += dt * (1.f-alpha) * sample.w;
    }
  }
  return t;
}

struct HitRecordVolume
{
  bool hit{false};
  unsigned volID{UINT_MAX};
  unsigned fieldID{UINT_MAX};
  float t{FLT_MAX};
  float3 albedo{0.f,0.f,0.f};
  float extinction{0.f};
  float Tr{1.f};
  int inst_id{-1};
};

VSNRAY_FUNC
inline HitRecordVolume sampleFreeFlightDistance(
    ScreenSample &ss, Ray ray, const dco::Volume &vol,
    VisionarayGlobalState::DeviceObjectRegistry onDevice) {

  HitRecordVolume hr;
  hr.t = ray.tmax;

  const float dt = onDevice.spatialFields[vol.asTransferFunction1D.fieldID].baseDT;

  dco::GridAccel grid
      = onDevice.spatialFields[vol.asTransferFunction1D.fieldID].gridAccel;

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {
#ifdef WITH_CUDA
    const bool hasMajorant = false;
#else
    const bool hasMajorant =
        onDevice.spatialFields[vol.asTransferFunction1D.fieldID].type
            == dco::SpatialField::Unstructured ||
        onDevice.spatialFields[vol.asTransferFunction1D.fieldID].type
            == dco::SpatialField::StructuredRegular;
#endif
    const float majorant = hasMajorant ? grid.maxOpacities[leafID] : 1.f;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= logf(1.f - ss.random()) / majorant * dt;

      if (t >= t1)
        break;

      float3 P = ray.ori+ray.dir*t;
      float v = 0.f;
      if (sampleField(onDevice.spatialFields[vol.asTransferFunction1D.fieldID],P,v)) {
        float4 sample
            = postClassify(ss,onDevice.transferFunctions[vol.asTransferFunction1D.tfID],v);
        hr.albedo = sample.xyz();
        hr.extinction = sample.w;
        float u = ss.random();
        if (hr.extinction >= u * majorant) {
          hr.hit = true;
          hr.volID = vol.volID;
          hr.fieldID = vol.asTransferFunction1D.fieldID;
          hr.Tr = 0.f;
          hr.t = t;
          return false; // stop traversal
        }
      }
    }

    return true; // cont. traversal to the next spat. partition

  };

  auto boxHit = intersect(ray, vol.bounds);
  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);
#ifndef WITH_CUDA
  if (onDevice.spatialFields[vol.asTransferFunction1D.fieldID].type == dco::SpatialField::Unstructured ||
      onDevice.spatialFields[vol.asTransferFunction1D.fieldID].type == dco::SpatialField::StructuredRegular)
    dda3(ray, grid.dims, grid.worldBounds, woodcockFunc);
  else
    woodcockFunc(-1, ray.tmin, ray.tmax);
#else
    woodcockFunc(-1, ray.tmin, ray.tmax);
#endif

  return hr;
}

VSNRAY_FUNC
inline HitRecordVolume sampleFreeFlightDistanceAllVolumes(
    ScreenSample &ss, Ray ray, unsigned worldID,
    VisionarayGlobalState::DeviceObjectRegistry onDevice) {
  // find closest distance across all volumes
  HitRecordVolume result;
  result.t = ray.tmax;

  while (true) {
    auto hrv = intersectVolumes(ray, onDevice.TLSs[worldID]);
    if (!hrv.hit)
      break;
    const auto &inst = onDevice.instances[hrv.inst_id];
    const auto &group = onDevice.groups[inst.groupID];
    const auto &geom = onDevice.geometries[group.geoms[hrv.geom_id]];
    const auto &vol = geom.asVolume.data;
    HitRecordVolume hr = sampleFreeFlightDistance(ss, ray, vol, onDevice);
    if (hr.t < result.t) {
      result = hr;
      result.inst_id = hrv.inst_id;
    }
    auto boxHit = intersect(ray, vol.bounds);
    ray.tmin = boxHit.tfar + onDevice.worldEPS[worldID];
  }

  return result;
}

} // namespace visionaray
