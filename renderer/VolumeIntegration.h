// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "scene/volume/spatial_field/Connectivity.h"
#include "scene/volume/spatial_field/Plane.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

//=========================================================
// UElem marching helpers
//=========================================================

VSNRAY_FUNC
inline
void clip(const Ray ray,
          int &outID,
          float &out_t,
          const Plane &plane,
          int neighborID)
{
  float d = dot(plane.N,ray.dir);
  if (d >= 0.f) return;
  float t = plane.eval(ray.ori) / -d;
  if (t < out_t) {
    out_t = t;
    outID = neighborID;
  }
}

VSNRAY_FUNC
inline void evalTet(float3 P, const float4 v[8], float &value_out) {
  const float3 va = v[0].xyz();
  const float3 vb = v[1].xyz();
  const float3 vc = v[2].xyz();
  const float3 vd = v[3].xyz();

  const Plane pa = makePlane(vb,vd,vc);
  const Plane pb = makePlane(va,vc,vd);
  const Plane pc = makePlane(va,vd,vb);
  const Plane pd = makePlane(va,vb,vc);

  const float fa = pa.eval(P)/pa.eval(va);
  const float fb = pb.eval(P)/pb.eval(vb);
  const float fc = pc.eval(P)/pc.eval(vc);
  const float fd = pd.eval(P)/pd.eval(vd);

  value_out = fa*v[0].w + fb*v[1].w + fc*v[2].w + fd*v[3].w;
}

VSNRAY_FUNC
inline void evalPyr(float3 P, const float4 v[8], float &value_out) {
  const float f0 = v[0].w;
  const float f1 = v[1].w;
  const float f2 = v[2].w;
  const float f3 = v[3].w;
  const float f4 = v[4].w;

  const float3 p0 = v[0].xyz();
  const float3 p1 = v[1].xyz();
  const float3 p2 = v[2].xyz();
  const float3 p3 = v[3].xyz();
  const float3 p4 = v[4].xyz();

  const Plane base = makePlane(p0,p1,p2);
  float w = base.eval(P)/base.eval(p4);

  float u0 = makePlane(p0,p4,p1).eval(P);
  float u1 = makePlane(p2,p4,p3).eval(P);
  float u = u0 / max(u0+u1,1e-10f);

  float v0 = makePlane(p0,p3,p4).eval(P);
  float v1 = makePlane(p1,p4,p2).eval(P);
  float vv = v0 / max(v0+v1,1e-10f);

  value_out = w*f4 + (1.f-w)*((1.f-u)*(1.f-vv)*f0+
                              (1.f-u)*(    vv)*f1+
                              (    u)*(1.f-vv)*f3+
                              (    u)*(    vv)*f2);
}

VSNRAY_FUNC
inline void evalWedge(float3 P, const float4 v[8], float &value_out) {
  const float f0 = v[0].w;
  const float f1 = v[1].w;
  const float f2 = v[2].w;
  const float f3 = v[3].w;
  const float f4 = v[4].w;
  const float f5 = v[5].w;
  const float3 p0 = v[0].xyz();
  const float3 p1 = v[1].xyz();
  const float3 p2 = v[2].xyz();
  const float3 p3 = v[3].xyz();
  const float3 p4 = v[4].xyz();
  const float3 p5 = v[5].xyz();

  const Plane base = makePlane(p0,p1,p3);

  float w0 = base.eval(P);
  Plane top;
  top.N = cross(cross(base.N,p5-p2),p5-p2);
  top.d = dot(top.N,p2);
  const float w1 = top.eval(P);
  const float w = w0/(w0+w1+1e-10f);

  const Plane front = makePlane(p0,p2,p1);
  const Plane back  = makePlane(p3,p4,p5);
  const float u0 = front.eval(P);
  const float u1 = back.eval(P);
  const float u = u0/(u0+u1+1e-10f);

  const Plane left = makePlane(p0,p3,p2);
  const Plane right  = makePlane(p1,p2,p4);
  const float v0 = left.eval(P);
  const float v1 = right.eval(P);
  const float vv = v0/(v0+v1+1e-10f);

  const float fbase
    = (1.f-u)*(1.f-vv)*f0
    + (1.f-u)*(    vv)*f1
    + (    u)*(1.f-vv)*f3
    + (    u)*(    vv)*f4;
  const float ftop = (1.f-u)*f2 + u*f5;

  value_out = (1.f-w)*fbase + w*ftop;
}

VSNRAY_FUNC
inline void evalHex(float3 P, const float4 v[8], float &value_out) {
  const float3 v0 = v[0].xyz();
  const float3 v1 = v[1].xyz();
  const float3 v2 = v[2].xyz();
  const float3 v3 = v[3].xyz();
  const float3 v4 = v[4].xyz();
  const float3 v5 = v[5].xyz();
  const float3 v6 = v[6].xyz();
  const float3 v7 = v[7].xyz();

  const Plane frt = makePlane(v0,v4,v1);
  const Plane bck = makePlane(v3,v2,v7);
  const Plane lft = makePlane(v0,v3,v4);
  const Plane rgt = makePlane(v1,v5,v2);
  const Plane top = makePlane(v4,v7,v5);
  const Plane btm = makePlane(v0,v1,v3);

  const float t_frt = frt.eval(P);
  const float t_bck = bck.eval(P);
  const float t_lft = lft.eval(P);
  const float t_rgt = rgt.eval(P);
  const float t_top = top.eval(P);
  const float t_btm = btm.eval(P);

  const float f_x = t_lft/(t_lft+t_rgt);
  const float f_y = t_frt/(t_frt+t_bck);
  const float f_z = t_btm/(t_btm+t_top);
  
  float f0 = v[0].w;
  float f1 = v[1].w;
  float f2 = v[2].w;
  float f3 = v[3].w;
  float f4 = v[4].w;
  float f5 = v[5].w;
  float f6 = v[6].w;
  float f7 = v[7].w;
  value_out// = 1.f/8.f *(f0+f1+f2+f3+f4+f5+f6+f7);
    =
    + (1.f-f_z)*(1.f-f_y)*(1.f-f_x)*f0
    + (1.f-f_z)*(1.f-f_y)*(    f_x)*f1
    + (1.f-f_z)*(    f_y)*(1.f-f_x)*f3
    + (1.f-f_z)*(    f_y)*(    f_x)*f2
    + (    f_z)*(1.f-f_y)*(1.f-f_x)*f4
    + (    f_z)*(1.f-f_y)*(    f_x)*f5
    + (    f_z)*(    f_y)*(1.f-f_x)*f7
    + (    f_z)*(    f_y)*(    f_x)*f6
    ;
}

VSNRAY_FUNC
inline float evalElem(float3 P,
                      const conn::UElem &cElem,
                      size_t numVerts,
                      float cellValue)
{
  if (isnan(cElem.vertices[0].w))
    return cellValue;

  float value = 0.f;

  if (numVerts == 4)
    evalTet(P,cElem.vertices,value);
  else if (numVerts == 5)
    evalPyr(P,cElem.vertices,value);
  else if (numVerts == 6)
    evalWedge(P,cElem.vertices,value);
  else if (numVerts == 8)
    evalHex(P,cElem.vertices,value);
  else
    assert(0);

  return value;
}

VSNRAY_FUNC
inline void nextElem(const Ray &ray,
                     const conn::UElem &cElem,
                     size_t numVerts,
                     const uint64_t *faceNeighbors,
                     uint64_t currID,
                     uint64_t &outID,
                     float &out_t)
{
  int planeID = -1; // in [0:6)

  out_t = FLT_MAX;
  for (int i=0; i<cElem.numFaces(); ++i) {
    const conn::Face f = cElem.face(i);
    const Plane p = makePlane(f.vertex(0).xyz(),
                              f.vertex(1).xyz(),
                              f.vertex(2).xyz());
    clip(ray,planeID,out_t,p,i);
  }

  assert(planeID>=0 && planeID<6);
  outID = faceNeighbors[currID*6+planeID];
}

//=========================================================
// UElem marching main function
//=========================================================

template <bool Shading>
VSNRAY_FUNC
inline float elementMarchVolume(ScreenSample &ss,
                                const DeviceObjectRegistry &onDevice,
                                Ray ray,
                                const dco::Volume &vol,
                                const dco::LightRef *allLights,
                                unsigned numLights,
                                float3 ambientColor,
                                float ambientRadiance,
                                float samplingRateInv,
                                float3 &color,
                                float &alpha) {
  const auto &sf = vol.field;

  assert(sf.type == dco::SpatialField::Unstructured);

  float t=FLT_MAX;
  float3 viewDir = -ray.dir;
  const float alphaMax=0.99f;
  float transmittance = 1.f;
  while (alpha<alphaMax) {
    default_intersector isect;
#if defined(WITH_CUDA) || defined(WITH_HIP) || defined(WITH_SYCL)
    auto hr = intersect_rayN_bvh2<detail::ClosestHit>(ray,
                                                      sf.asUnstructured.shellBVH,
                                                      isect);
#else
    auto hr = intersect_ray1_bvhN<detail::ClosestHit>(ray,
                                                      sf.asUnstructured.shellBVH,
                                                      isect);
#endif

    if (!hr.hit || hr.t>=ray.tmax) break;

    const auto *triangles = sf.asUnstructured.shellBVH.primitives();
    auto tri = triangles[hr.prim_id];
    auto n = cross(tri.e1,tri.e2);

    struct {
      float t;
      unsigned elemID;
    } entry, exit;

    const bool frontFace = dot(viewDir,n) > 0.f;

    const float3 hitPos = ray.ori + hr.t * ray.dir;
    const float eps = epsilonFrom(hitPos, ray.dir, hr.t);

    Ray ray2 = ray;
    ray2.ori = hitPos - n * eps;
    ray2.tmin = 0.f;

    if (!frontFace) {
      // we're inside so have to search backwards
      // to find the entry position:
      ray2.dir *= -1.f;
    }

#if defined(WITH_CUDA) || defined(WITH_HIP) || defined(WITH_SYCL)
    auto hr2 = intersect_rayN_bvh2<detail::ClosestHit>(ray2,
                                                       sf.asUnstructured.shellBVH,
                                                       isect);
#else
    auto hr2 = intersect_ray1_bvhN<detail::ClosestHit>(ray2,
                                                       sf.asUnstructured.shellBVH,
                                                       isect);
#endif

    if (!hr2.hit) break;

    if (frontFace) {
      entry.t = hr.t;
      entry.elemID = hr.geom_id;

      exit.t = hr.t+hr2.t;
      exit.elemID = hr2.geom_id;
    } else {
      entry.t = hr.t-hr2.t;
      entry.elemID = hr2.geom_id;

      exit.t = hr.t;
      exit.elemID = hr.geom_id;
    }

    t = entry.t;
    uint64_t elemID = entry.elemID;
    uint64_t nextID;
    interval<float> currentRange(entry.t,FLT_MAX);
    dco::UElem elem = sf.asUnstructured.elems[elemID];
    size_t numVerts = elem.end-elem.begin;
    conn::UElem cElem(elem);
    nextElem(ray,cElem,numVerts,sf.asUnstructured.faceNeighbors,elemID,nextID,currentRange.max);

    float dt = sf.cellSize*samplingRateInv;

    while (t < exit.t && alpha<alphaMax) {
      while (!currentRange.contains(t)) {
        elemID = nextID;

        if (elemID == ~0ull)
          break;

        elem = sf.asUnstructured.elems[elemID];
        numVerts = elem.end-elem.begin;
        cElem = conn::UElem(elem);
        currentRange.min = currentRange.max;
        nextElem(ray,cElem,numVerts,sf.asUnstructured.faceNeighbors,elemID,nextID,currentRange.max);
      }

      if (elemID == ~0ull)
        break;

      if (t > ray.tmin) {
        float3 P = ray.ori+ray.dir*t;
        float value = evalElem(P,cElem,numVerts,elem.cellValue);
        float4 sample = postClassify(vol.asTransferFunction1D,value);

        float3 shadedColor = sample.xyz();
        float stepTransmittance = powf(1.f - sample.w, min(dt,exit.t-t) / vol.unitDistance);
        color += transmittance * (1.f - stepTransmittance) * shadedColor;
        alpha += transmittance * (1.f - stepTransmittance);
        transmittance *= stepTransmittance;
      }

      t += dt;
    }

    const float3 exitPos = ray.ori + exit.t * ray.dir;
    const float epsExit = epsilonFrom(exitPos, ray.dir, exit.t);
    ray.ori = exitPos + n * epsExit;
  }
  return t;
}

//=========================================================
// Ray marching all other field types
//=========================================================

template <bool Shading>
VSNRAY_FUNC
inline float rayMarchVolume(ScreenSample &ss,
                            const DeviceObjectRegistry &onDevice,
                            Ray ray,
                            const dco::Volume &vol,
                            const dco::LightRef *allLights,
                            unsigned numLights,
                            float3 ambientColor,
                            float ambientRadiance,
                            float samplingRateInv,
                            float3 &color,
                            float &alpha) {
  const auto &sf = vol.field;

  auto boxHit = intersect(ray, vol.bounds);

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  float dt = sf.cellSize*samplingRateInv;

  float3 delta(sf.cellSize, sf.cellSize, sf.cellSize);
  delta *= float3(sf.voxelSpaceTransform(0,0),
                  sf.voxelSpaceTransform(1,1),
                  sf.voxelSpaceTransform(2,2));

  float3 viewDir = -ray.dir;

  float t=ray.tmin;
  float transmittance = 1.f;
  for (;t<=ray.tmax&&alpha<0.99f;t+=dt) {
    float3 P = ray.ori+ray.dir*t;
    float v = 0.f;
    if (sampleField(sf,P,v)) {
      float4 sample
          = postClassify(vol.asTransferFunction1D,v);
      float stepTransmittance =
          powf(1.f - sample.w, dt / vol.unitDistance);

      // Gradient shading:
      float3 shadedColor = sample.xyz();
      if constexpr (Shading) {
        float3 gn(0.f);
        if (sampleGradient(sf,P,delta,gn))
          gn = normalize(gn);

        gn = faceforward(gn, viewDir, gn);

        shadedColor = float3(0.f);

        if (length(gn) > 1e-10f) {
          auto safe_rcp = [](float f) { return f > 0.f ? 1.f/f : 0.f; };
          for (unsigned lightID=0; lightID<numLights; ++lightID) {
            const dco::Light &light = getLight(allLights, lightID, onDevice);

            LightSample ls = sampleLight(light, P, ss.random);

            dco::Material mat = dco::createMaterial();
            mat.type = dco::Material::Matte;
            mat.asMatte.color = dco::createMaterialParamRGB();
            mat.asMatte.color.rgb = sample.xyz();

            auto color = evalMaterial(mat,
                                      onDevice,
                                      {}, // attribs, not used..
                                      float3(0.f), // objPos, not used..
                                      UINT_MAX, // primID, not used..
                                      gn, gn,
                                      float3(0.f), float3(0.f), // tangent, bitangent
                                      normalize(viewDir),
                                      normalize(ls.dir),
                                      ls.intensity * safe_rcp(ls.dist2));
            color = color * safe_rcp(ls.pdf);
            shadedColor += color;
          }
        }

        shadedColor += sample.xyz() * ambientColor * ambientRadiance;
      }

      color += transmittance * (1.f - stepTransmittance) * shadedColor;
      alpha += transmittance * (1.f - stepTransmittance);
      transmittance *= stepTransmittance;
    }
  }
  return t;
}

} // namespace visionaray
