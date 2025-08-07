#pragma once

#include "renderer/common.h"
#include "renderer/DDA.h"
#include "scene/volume/spatial_field/Connectivity.h"
#include "scene/volume/spatial_field/Plane.h"
#include "DeviceCopyableObjects.h"

namespace visionaray {

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
inline void evalTet(float3 P, const Plane p[6], const float4 v[8], float &value_out) {
  float3 va = v[0].xyz();
  float3 vb = v[1].xyz();
  float3 vc = v[2].xyz();
  float3 vd = v[3].xyz();
  const float fa = p[0].eval(P)/p[0].eval(va);
  const float fb = p[1].eval(P)/p[1].eval(vb);
  const float fc = p[2].eval(P)/p[2].eval(vc);
  const float fd = p[3].eval(P)/p[3].eval(vd);

  value_out = fa*v[0].w + fb*v[1].w + fc*v[2].w + fd*v[3].w;
}

VSNRAY_FUNC
inline void evalPyr(float3 P, const Plane p[6], const float4 v[8], float &value_out) {
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

  float w = p[0].eval(P)/p[0].eval(p4);

  float u0 = p[1].eval(P);
  float u1 = p[2].eval(P);
  float u = u0 / max(u0+u1,1e-10f);

  float v0 = p[3].eval(P);
  float v1 = p[4].eval(P);
  float vv = v0 / max(v0+v1,1e-10f);

  value_out = w*f4 + (1.f-w)*((1.f-u)*(1.f-vv)*f0+
                              (1.f-u)*(    vv)*f1+
                              (    u)*(1.f-vv)*f3+
                              (    u)*(    vv)*f2);
}

VSNRAY_FUNC
inline void evalWedge(float3 P, const Plane p[6], const float4 v[8], float &value_out) {
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

  float w = p[0].eval(P);

  float u0 = p[1].eval(P);
  float u1 = p[2].eval(P);
  float u = u0 / max(u0+u1,1e-10f);

  float v0 = p[3].eval(P);
  float v1 = p[4].eval(P);
  float vv = v0 / max(v0+v1,1e-10f);

  const float fbase
    = (1.f-u)*(1.f-vv)*f0
    + (1.f-u)*(    vv)*f1
    + (    u)*(1.f-vv)*f3
    + (    u)*(    vv)*f4;
  const float ftop = (1.f-u)*f2 + u*f5;
  value_out = (1.f-w)*fbase + w*ftop;
}

VSNRAY_FUNC
inline void evalHex(float3 P, const Plane p[6], const float4 v[8], float &value_out) {
  const float t_frt = p[0].eval(P); //if (t_frt < 0.f) return false;
  const float t_bck = p[1].eval(P); //if (t_bck < 0.f) return false;
  const float t_lft = p[2].eval(P); //if (t_lft < 0.f) return false;
  const float t_rgt = p[3].eval(P); //if (t_rgt < 0.f) return false;
  const float t_top = p[4].eval(P); //if (t_top < 0.f) return false;
  const float t_btm = p[5].eval(P); //if (t_btm < 0.f) return false;

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
                      const Plane p[6],
                      size_t numVerts,
                      float cellValue)
{
  if (isnan(cElem.vertices[0].w))
    return cellValue;

  float value = 0.f;

  if (numVerts == 4)
    evalTet(P,p,cElem.vertices,value);
  else if (numVerts == 5)
    evalPyr(P,p,cElem.vertices,value);
  else if (numVerts == 6)
    evalWedge(P,p,cElem.vertices,value);
  else if (numVerts == 8)
    evalHex(P,p,cElem.vertices,value);
  else
    assert(0);

  return value;
}

VSNRAY_FUNC
inline void nextElem(const Ray &ray,
                     const conn::UElem &cElem,
                     const Plane p[6],
                     size_t numVerts,
                     const uint64_t *faceNeighbors,
                     uint64_t currID,
                     uint64_t &outID,
                     float &out_t)
{
  int planeID = -1; // in [0:6)

  out_t = FLT_MAX;
  for (int i=0; i<cElem.numFaces(); ++i) {
    clip(ray,planeID,out_t,p[i],i);
  }

  assert(planeID>=0 && planeID<6);
  outID = faceNeighbors[currID*6+planeID];
}

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
#if defined(WITH_CUDA) || defined(WITH_HIP)
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
    unsigned entry_id, exit_id;
    if (dot(viewDir,n) > 0.f) {
      // front face hit, find exit face:
      Ray ray2 = ray;
      const float3 hitPos = ray.ori + hr.t * ray.dir;
      const float eps = epsilonFrom(hitPos, ray.dir, hr.t);
      ray2.tmin = hr.t + eps;

#if defined(WITH_CUDA) || defined(WITH_HIP)
      auto hr2 = intersect_rayN_bvh2<detail::ClosestHit>(ray2,
                                                         sf.asUnstructured.shellBVH,
                                                         isect);
#else
      auto hr2 = intersect_ray1_bvhN<detail::ClosestHit>(ray2,
                                                         sf.asUnstructured.shellBVH,
                                                         isect);
#endif

      entry.t = hr.t;
      entry.elemID = hr.geom_id;

      exit.t = hr2.t;
      exit.elemID = hr2.geom_id;
    } else {
      // back face hit, find entry face:
      Ray ray2 = ray;
      const float3 hitPos = ray.ori + hr.t * ray.dir;
      const float eps = epsilonFrom(hitPos, ray.dir, hr.t);
      ray2.tmin = hr.t + eps;
      ray2.dir *= -1.f;

#if defined(WITH_CUDA) || defined(WITH_HIP)
      auto hr2 = intersect_rayN_bvh2<detail::ClosestHit>(ray2,
                                                         sf.asUnstructured.shellBVH,
                                                         isect);
#else
      auto hr2 = intersect_ray1_bvhN<detail::ClosestHit>(ray2,
                                                         sf.asUnstructured.shellBVH,
                                                         isect);
#endif

      entry.t = hr2.t;
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
    Plane p[6];
    for (int i=0; i<cElem.numFaces(); ++i) {
      const conn::Face f = cElem.face(i);
      p[i] = makePlane(f.vertex(0).xyz(),
                       f.vertex(1).xyz(),
                       f.vertex(2).xyz());
    }
    nextElem(ray,cElem,p,numVerts,sf.asUnstructured.faceNeighbors,elemID,nextID,currentRange.max);

    float dt = sf.cellSize*samplingRateInv;

    while (t < exit.t && alpha<alphaMax) {
      while (!currentRange.contains(t)) {
        elemID = nextID;

        if (elemID == ~0ull)
          break;

        elem = sf.asUnstructured.elems[elemID];
        cElem = conn::UElem(elem);
        for (int i=0; i<cElem.numFaces(); ++i) {
          const conn::Face f = cElem.face(i);
          p[i] = makePlane(f.vertex(0).xyz(),
                           f.vertex(1).xyz(),
                           f.vertex(2).xyz());
        }
        currentRange.min = currentRange.max;
        nextElem(ray,cElem,p,numVerts,sf.asUnstructured.faceNeighbors,elemID,nextID,currentRange.max);
      }

      if (elemID == ~0ull)
        break;

      if (t > ray.tmin) {
        float3 P = ray.ori+ray.dir*t;
        float value = evalElem(P,cElem,p,numVerts,elem.cellValue);
        float4 sample = postClassify(vol.asTransferFunction1D,value);

        float3 shadedColor = sample.xyz();
        float stepTransmittance = powf(1.f - sample.w, dt / vol.unitDistance);
        color += transmittance * (1.f - stepTransmittance) * shadedColor;
        alpha += transmittance * (1.f - stepTransmittance);
        transmittance *= stepTransmittance;
      }

      t += dt;
    }

    const float3 exitPos = ray.ori + exit.t * ray.dir;
    const float eps = epsilonFrom(exitPos, ray.dir, exit.t);
    ray.tmin = exit.t + eps;
  }
  return t;
}

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

  // special case: element marching
  if (sf.type == dco::SpatialField::Unstructured) {
    return elementMarchVolume<Shading>(ss,
                                       onDevice,
                                       ray,
                                       vol,
                                       allLights,
                                       numLights,
                                       ambientColor,
                                       ambientRadiance,
                                       samplingRateInv,
                                       color,
                                       alpha);
  }

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
  for (;t<ray.tmax&&alpha<0.99f;t+=dt) {
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
                                      nullptr, // attribs, not used..
                                      float3(0.f), // objPos, not used..
                                      UINT_MAX, // primID, not used..
                                      gn, gn,
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
