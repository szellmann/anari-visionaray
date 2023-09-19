#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

VSNRAY_FUNC
inline bool sampleField(dco::SpatialField sf, vec3 P, float &value) {
  if (sf.type == dco::SpatialField::StructuredRegular) {
    value = tex3D(sf.asStructuredRegular.sampler,
        sf.asStructuredRegular.objectToTexCoord(P));
    return true;
  } else if (sf.type == dco::SpatialField::Unstructured) {
    Ray ray;
    ray.ori = P;
    ray.dir = float3(1.f);
    ray.tmin = ray.tmax = 0.f;
    auto hr = intersect(ray, sf.asUnstructured.samplingBVH);

    if (!hr.hit)
      return false;

    value = hr.u; // value is stored in "u"!
    return true;
  }

  return false;
}

VSNRAY_FUNC
inline bool sampleGradient(dco::SpatialField sf, vec3 P, float3 &value) {
  float x0=0, x1=0, y0=0, y1=0, z0=0, z1=0;
  bool b0 = sampleField(sf, P+float3{sf.baseDT, 0.f, 0.f}, x1);
  bool b1 = sampleField(sf, P-float3{sf.baseDT, 0.f, 0.f}, x0);
  bool b2 = sampleField(sf, P+float3{0.f, sf.baseDT, 0.f}, y1);
  bool b3 = sampleField(sf, P-float3{0.f, sf.baseDT, 0.f}, y0);
  bool b4 = sampleField(sf, P+float3{0.f, 0.f, sf.baseDT}, z1);
  bool b5 = sampleField(sf, P-float3{0.f, 0.f, sf.baseDT}, z0);
  if (b0 && b1 && b2 && b3 && b4 && b5) {
    value = float3{x1,y1,z1}-float3{x0,y0,z0};
    return true; // TODO
  } else {
    value = float3{0.f};
    return false;
  }
}

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
  float dt = onDevice.spatialFields[vol.fieldID].baseDT;
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
    if (sampleField(onDevice.spatialFields[vol.fieldID],P,v)) {
      float4 sample
          = postClassify(ss,onDevice.transferFunctions[vol.volID],v);
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
};

VSNRAY_FUNC
inline HitRecordVolume sampleFreeFlightDistance(
    ScreenSample &ss, Ray ray, const dco::Volume &vol,
    VisionarayGlobalState::DeviceObjectRegistry onDevice) {

  HitRecordVolume hr;
  hr.t = ray.tmax;

  const float dt = onDevice.spatialFields[vol.fieldID].baseDT;

  dco::GridAccel grid = onDevice.gridAccels[vol.fieldID];

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {
    const float majorant
        = onDevice.spatialFields[vol.fieldID].type == dco::SpatialField::Unstructured ? grid.maxOpacities[leafID] : 1.f;
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= logf(1.f - ss.random()) / majorant * dt;

      if (t >= t1)
        break;

      float3 P = ray.ori+ray.dir*t;
      float v = 0.f;
      if (sampleField(onDevice.spatialFields[vol.fieldID],P,v)) {
        float4 sample
            = postClassify(ss,onDevice.transferFunctions[vol.volID],v);
        hr.albedo = sample.xyz();
        hr.extinction = sample.w;
        float u = ss.random();
        if (hr.extinction >= u * majorant) {
          hr.hit = true;
          hr.volID = vol.volID;
          hr.fieldID = vol.fieldID;
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
  if (onDevice.spatialFields[vol.fieldID].type == dco::SpatialField::Unstructured)
    dda3(ray, grid.dims, grid.worldBounds, woodcockFunc);
  else
    woodcockFunc(-1, ray.tmin, ray.tmax);

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
    auto inst = onDevice.instances[hrv.inst_id];
    const auto &geom = onDevice.groups[inst.groupID].geoms[hrv.geom_id];
    const auto &vol = geom.asVolume.data;
    HitRecordVolume hr = sampleFreeFlightDistance(ss, ray, vol, onDevice);
    if (hr.t < result.t) {
      result = hr;
    }
    auto boxHit = intersect(ray, vol.bounds);
    ray.tmin = boxHit.tfar + 1e-3f;
  }

  return result;
}

} // namespace visionaray
