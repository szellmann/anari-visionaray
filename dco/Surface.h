// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dco/common.h"
#include "dco/DDA.h"
#include "dco/SpatialField.h"

namespace visionaray::dco {

// Surface and geometry attributes //

enum class Attribute
{
  _0,
  _1,
  _2,
  _3,
  Color,
  WorldPos,
  WorldNormal,
  ObjectPos,
  ObjectNormal,
  None,
};

struct AttributeRec
{
  float4 _0;
  float4 _1;
  float4 _2;
  float4 _3;
  float4 color;
  float4 worldPos;
  float4 worldNormal;
  float4 objectPos;
  float4 objectNormal;

  VSNRAY_FUNC
  inline float4 get(Attribute attr) const
  {
    switch (attr) {
    case Attribute::_0: return _0;
    case Attribute::_1: return _1;
    case Attribute::_2: return _2;
    case Attribute::_3: return _3;
    case Attribute::Color: return color;
    case Attribute::WorldPos: return worldPos;
    case Attribute::WorldNormal: return worldNormal;
    case Attribute::ObjectPos: return objectPos;
    case Attribute::ObjectNormal: return objectNormal;
    case Attribute::None: default: break; // fall-through to function return
    }
    return float4(0,0,0,1);
  }
};

// ISO surface //

struct ISOSurface
{
  unsigned isoID{UINT_MAX};
  unsigned geomID{UINT_MAX};

  SpatialField field;
  unsigned numValues{0};
  const float *values{nullptr};

  aabb bounds;
};

VSNRAY_FUNC
inline aabb get_bounds(const ISOSurface &iso)
{
  return iso.bounds;
}

inline void split_primitive(
    aabb &L, aabb &R, float plane, int axis, const ISOSurface &vol)
{
  assert(0);
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    Ray ray, const ISOSurface &iso)
{
  hit_record<Ray, primitive<unsigned>> result;

  auto boxHit = intersect(ray, iso.bounds);
  if (!boxHit.hit)
    return result;

  const auto &sf = iso.field;

  float unitDistance = 1.f;

  auto isectFunc = [&](const int leafID, float t0, float t1) {
    bool empty = (leafID != -1);

    float dt = unitDistance * sf.gridAccel.stepSize(leafID);
    box1 valueRange = sf.gridAccel.valueRange(leafID);

    for (unsigned i=0;i<iso.numValues;i++) {
      float isoValue = iso.values[i];
      if (valueRange.contains(isoValue)) {
        empty = false;
        break;
      }
    }

    if (empty)
      return true;

    float t0_old = t0;
    float t1_old = t1;
    t0 = t1 = ray.tmin-dt/2.f;
    while (t0 < t0_old) t0 += dt;
    while (t1 < t1_old) t1 += dt;

    float3 P1 = ray.ori+ray.dir*t0;
    float v1 = 0.f;
    bool sample1 = sampleField(sf,P1,v1);

    for (float t=t0;t<t1;t+=dt) {
      float3 P2 = ray.ori+ray.dir*(t+dt);
      float v2 = 0.f;
      bool sample2 = sampleField(sf,P2,v2);
      if (sample1 && sample2) {
        box1f ival(fminf(v1,v2), fmaxf(v1,v2));
        unsigned numISOs = iso.numValues;
        bool hit=false;
        for (unsigned i=0;i<numISOs;i++) {
          float isoValue = iso.values[i];
          if (ival.contains(isoValue)) {
            //float tHit = t+dt/2.f;
            float f = (isoValue-v1) / (v2-v1);
            float tHit = t+dt*f;
            if (tHit < result.t) {
              result.hit = true;
              result.prim_id = i;
              result.geom_id = iso.geomID;
              result.t = tHit;
            }
            hit = true;
          }
        }
        if (hit) return false; // stop traversal
      }
      P1 = P2;
      v1 = v2;
      sample1 = sample2;
    }

    return true; // cont. traversal to the next spat. partition
  };

  ray.tmin = max(ray.tmin, boxHit.tnear);
  ray.tmax = min(ray.tmax, boxHit.tfar);

  // transform ray to voxel space
  ray.ori = sf.pointToVoxelSpace(ray.ori);
  ray.dir = sf.vectorToVoxelSpace(ray.dir);

  const float dt_scale = length(ray.dir);
  ray.dir = normalize(ray.dir);

  ray.tmin = ray.tmin * dt_scale;
  ray.tmax = ray.tmax * dt_scale;
  unitDistance = unitDistance * dt_scale;

  if (sf.gridAccel.isValid())
    dda3(ray, sf.gridAccel.dims, sf.gridAccel.gridBounds, isectFunc);
  else
    isectFunc(-1, boxHit.tnear, boxHit.tfar);

  if (result.hit) {
    result.t /= dt_scale;
  }

  return result;
}

// Cone primitive //

struct Cone : public primitive<unsigned>
{
  float3 v1, v2;
  float r1, r2;
};

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &r, const Cone &cone)
{
  // From https://iquilezles.org/articles/intersectors/
  hit_record<Ray, primitive<unsigned>> result;
  result.hit = false;

  const vec3f &ro = r.ori;
  const vec3f &rd = r.dir;

  const vec3f &pa = cone.v1;
  const vec3f &pb = cone.v2;

  const float ra = cone.r1;
  const float rb = cone.r2;

  const vec3f ba = pb - pa;
  const vec3f oa = ro - pa;
  const vec3f ob = ro - pb;
  const float m0 = dot(ba,ba);
  const float m1 = dot(oa,ba);
  const float m2 = dot(rd,ba);
  const float m3 = dot(rd,oa);
  const float m5 = dot(oa,oa);
  const float m9 = dot(ob,ba);

  auto dot2 = [](const vec3f v) { return dot(v,v); };

  // Caps:
  if (m1 < 0.f) {
    if (dot2(oa*m2-rd*m1) < ra*ra*m2*m2) {
      result.t = -m1 / m2;
      result.u = 0.f;
      result.hit = true;
      result.isect_pos = r.ori + result.t * r.dir;
      result.prim_id = cone.prim_id;
      result.geom_id = cone.geom_id;
      return result;
    }
  } else if (m9 > 0.f) {
    const float t = -m9/m2;
    if (dot2(ob+rd*t) < rb*rb) {
      result.t = t;
      result.u = 1.f;
      result.hit = true;
      result.isect_pos = r.ori + result.t * r.dir;
      result.prim_id = cone.prim_id;
      result.geom_id = cone.geom_id;
      return result;
    }
  }

  // Body
  const float rr = ra - rb;
  const float hy = m0 + rr*rr;
  const float k2 = m0*m0 - m2*m2*hy;
  const float k1 = m0*m0*m3 - m1*m2*hy + m0*ra*(rr*m2*1.f);
  const float k0 = m0*m0*m5 - m1*m1*hy + m0*ra*(rr*m1*2.f - m0*ra);
  const float h = k1*k1 - k2*k0;
  if (h < 0.f) return result;
  const float t = (-k1-sqrtf(h))/k2;
  const float y = m1 + t*m2;

  if (y > 0.f && y<m0) {
    result.t = t;
    result.u = y/m0;
    result.v = y;
    result.hit = true;
    result.isect_pos = r.ori + result.t * r.dir;
    result.prim_id = cone.prim_id;
    result.geom_id = cone.geom_id;
  }

  return result;
}

VSNRAY_FUNC inline aabb get_bounds(const Cone &cone)
{
  aabb result;
  result.invalidate();
  result.insert(cone.v1 - cone.r1);
  result.insert(cone.v1 + cone.r1);
  result.insert(cone.v2 - cone.r2);
  result.insert(cone.v2 + cone.r2);
  return result;
}

VSNRAY_FUNC inline void split_primitive(
    aabb& L, aabb& R, float plane, int axis, const Cone &cone)
{
  VSNRAY_UNUSED(L);
  VSNRAY_UNUSED(R);
  VSNRAY_UNUSED(plane);
  VSNRAY_UNUSED(axis);
  VSNRAY_UNUSED(cone);

  // TODO: implement this to support SBVHs
}

// Bezier curve primitive //

struct BezierCurve : public primitive<unsigned>
{
  float3 w0, w1, w2, w3;
  float r;

  VSNRAY_FUNC vec3 f(float t) const
  {
    float tinv = 1.0f - t;
    return tinv * tinv * tinv * w0
     + 3.0f * tinv * tinv * t * w1
        + 3.0f * tinv * t * t * w2
                  + t * t * t * w3;
  }

  VSNRAY_FUNC vec3 dfdt(float t) const
  {
    float tinv = 1.0f - t;
    return                 -3.0f * tinv * tinv * w0
     + 3.0f * (3.0f * t * t - 4.0f * t + 1.0f) * w1
                + 3.0f * (2.0f - 3.0f * t) * t * w2
                                + 3.0f * t * t * w3;
  }
};

VSNRAY_FUNC
inline BezierCurve make_bezierCurve(
    const vec3 &w0, const vec3 &w1, const vec3 &w2, const vec3 &w3, float r)
{
  BezierCurve curve;
  curve.w0 = w0;
  curve.w1 = w1;
  curve.w2 = w2;
  curve.w3 = w3;
  curve.r = r;
  return curve;
}

//=========================================================
// Phantom Ray-Hair Intersector (Reshetov and Luebke, 2018)
//=========================================================

namespace phantom {

// Ray/cone intersection from appendix A

struct RayConeIntersection
{
  VSNRAY_FUNC inline bool intersect(float r, float dr)
  {
    float r2  = r * r;
    float drr = r * dr;

    float ddd = cd.x * cd.x + cd.y * cd.y;
    dp        = c0.x * c0.x + c0.y * c0.y;
    float cdd = c0.x * cd.x + c0.y * cd.y;
    float cxd = c0.x * cd.y - c0.y * cd.x;

    float c = ddd;
    float b = cd.z * (drr - cdd);
    float cdz2 = cd.z * cd.z;
    ddd += cdz2;
    float a = 2.0f * drr * cdd + cxd * cxd - ddd * r2 + dp * cdz2;

    float discr = b * b - a * c;
    s   = (b - (discr > 0.0f ? sqrtf(discr) : 0.0f)) / c;
    dt  = (s * cd.z - cdd) / ddd;
    dc  = s * s + dp;
    sp  = cdd / cd.z;
    dp += sp * sp;

    return discr > 0.0f;
  }

  vec3  c0;
  vec3  cd;
  float s;
  float dt;
  float dp;
  float dc;
  float sp;
};

// TODO: use visionaray's ray/cyl test?!
VSNRAY_FUNC inline
bool intersectCylinder(const Ray &ray, vec3 p0, vec3 p1, float ra)
{
  vec3  ba = p1 - p0;
  vec3  oc = ray.ori - p0;

  float baba = dot(ba, ba);
  float bard = dot(ba, ray.dir);
  float baoc = dot(ba, oc);

  float k2 = baba - bard * bard;
  float k1 = baba * dot(oc, ray.dir) - baoc * bard;
  float k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;

  float h = k1 * k1 - k2 * k0;

  if (h < 0.0f)
    return false;

  h = sqrtf(h);
  float t = (-k1 - h) / k2;

  // body
  float y = baoc + t * bard;
  if (y > 0.0f && y < baba)
    return true;

  // caps
  t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;
  if (fabsf(k1 + k2 * t) < h)
    return true;

  return false;
}

struct TransformToRCC
{
  VSNRAY_FUNC inline TransformToRCC(const Ray &r)
  {
    vec3 e1;
    vec3 e2;
    vec3 e3 = normalize(r.dir);
    make_orthonormal_basis(e1, e2, e3);
    xformInv = mat4(
        vec4(e1,    0.0f),
        vec4(e2,    0.0f),
        vec4(e3,    0.0f),
        vec4(r.ori, 1.0f)
        );
    xform = inverse(xformInv);
  }

  VSNRAY_FUNC inline vec3 xfmPoint(vec3 point)
  { return (xform * vec4(point, 1.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmVector(vec3 vector)
  { return (xform * vec4(vector, 0.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmPointInv(vec3 point)
  { return (xformInv * vec4(point, 1.0f)).xyz(); }

  VSNRAY_FUNC inline vec3 xfmVectorInv(vec3 vector)
  { return (xformInv * vec4(vector, 0.0f)).xyz(); }

  mat4 xform;
  mat4 xformInv;
};

} // namespace phantom

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(
    const Ray &r, const BezierCurve &curve)
{
  hit_record<Ray, primitive<unsigned>> result;
  result.hit = false;

  // Early exit check against enclosing cylinder
  auto distToCylinder = [&curve](vec3 pt) {
    return length(cross(pt - curve.w0, pt - curve.w3)) / length(curve.w3 - curve.w0);
  };

  // TODO: could compute tighter bounding cylinder than this one!
  float rmax = distToCylinder(curve.f(0.33333f));
  rmax = fmaxf(rmax, distToCylinder(curve.f(0.66667f)));
  rmax += curve.r;

  vec3 axis = normalize(curve.w3 - curve.w0);
  vec3 p0   = curve.w0 - axis * curve.r;
  vec3 p1   = curve.w3 + axis * curve.r;

  if (!phantom::intersectCylinder(r, p0, p1, rmax))
    return result;

  // Transform curve to RCC
  phantom::TransformToRCC rcc(r);
  BezierCurve xcurve = make_bezierCurve(
      rcc.xfmPoint(curve.w0),
      rcc.xfmPoint(curve.w1),
      rcc.xfmPoint(curve.w2),
      rcc.xfmPoint(curve.w3),
      curve.r
      );

  // "Test for convergence. If the intersection is found,
  // report it, otherwise start at the other endpoint."

  // Compute curve end to start at
  float tstart = dot(xcurve.w3 - xcurve.w0, r.dir) > 0.0f ? 0.0f : 1.0f;

  for (int ep = 0; ep < 2; ++ep)
  {
    float t   = tstart;

    phantom::RayConeIntersection rci;

    float told = 0.0f;
    float dt1 = 0.0f;
    float dt2 = 0.0f;

    for (int i = 0; i < 40; ++i)
    {
      rci.c0 = xcurve.f(t);
      rci.cd = xcurve.dfdt(t);

      bool phantom = !rci.intersect(curve.r, 0.0f/*cylinder*/);

      // "In all examples in this paper we stop iterations when dt < 5x10^−5"
      if (!phantom && fabsf(rci.dt) < 5e-5f) {
        //vec3 n = normalize(curve.dfdt(t));
        rci.s += rci.c0.z;
        result.t = rci.s;
        result.u = t; // abuse param u to store curve's t
        result.hit = true;
        result.isect_pos = r.ori + result.t * r.dir;
        break;
      }

      rci.dt = min(rci.dt, 0.5f);
      rci.dt = max(rci.dt, -0.5f);

      dt1 = dt2;
      dt2 = rci.dt;

      // Regula falsi
      if (dt1 * dt2 < 0.0f) {
        float tnext = 0.0f;
        // "we use the simplest possible approach by switching
        // to the bisection every 4th iteration:"
        if ((i & 3) == 0)
          tnext = 0.5f * (told + t);
        else
          tnext = (dt2 * told - dt1 * t) / (dt2 - dt1);
        told = t;
        t = tnext;
      } else {
        told = t;
        t += rci.dt;
      }

      if (t < 0.0f || t > 1.0f)
        break;
    }

    if (!result.hit)
      tstart = 1.0f - tstart;
    else
      break;
  }

  return result;
}

// From here: https://www.shadertoy.com/view/MdKBWt
VSNRAY_FUNC inline aabb get_bounds(const BezierCurve &curve)
{
  vec3 p0 = curve.w0;
  vec3 p1 = curve.w1;
  vec3 p2 = curve.w2;
  vec3 p3 = curve.w3;

  // extremes
  vec3 mi = min(p0,p3);
  vec3 ma = max(p0,p3);

  // note pascal triangle coefficnets
  vec3 c = -1.0f*p0 + 1.0f*p1;
  vec3 b =  1.0f*p0 - 2.0f*p1 + 1.0f*p2;
  vec3 a = -1.0f*p0 + 3.0f*p1 - 3.0f*p2 + 1.0f*p3;

  // check if curve is quadratic, then derivative is a line.
  // in that case we'll just lazily insert the remaining control points..
  for (int d=0; d<3; ++d) {
    if (a[d] == 0.f) {
      mi[d] = min(mi[d],p1[d]);
      mi[d] = min(mi[d],p2[d]);
      ma[d] = max(mi[d],p1[d]);
      ma[d] = max(mi[d],p2[d]);
    }
  }

  vec3 h = b*b - a*c;

  // real solutions
  if (h.x > 0.0f || h.y > 0.0f || h.z > 0.0f)
  {
    vec3 g(sqrtf(fabsf(h.x)), sqrtf(fabsf(h.y)), sqrtf(fabsf(h.z)));
    vec3 t1 = clamp((-b - g)/a,vec3(0.0f),vec3(1.0f)); vec3 s1 = 1.0f-t1;
    vec3 t2 = clamp((-b + g)/a,vec3(0.0f),vec3(1.0f)); vec3 s2 = 1.0f-t2;
    vec3 q1 = s1*s1*s1*p0 + 3.0f*s1*s1*t1*p1 + 3.0f*s1*t1*t1*p2 + t1*t1*t1*p3;
    vec3 q2 = s2*s2*s2*p0 + 3.0f*s2*s2*t2*p1 + 3.0f*s2*t2*t2*p2 + t2*t2*t2*p3;

    if (h.x > 0.0f) {
      mi.x = min(mi.x,min(q1.x,q2.x));
      ma.x = max(ma.x,max(q1.x,q2.x));
    }

    if (h.y > 0.0f) {
      mi.y = min(mi.y,min(q1.y,q2.y));
      ma.y = max(ma.y,max(q1.y,q2.y));
    }

    if (h.z > 0.0f) {
      mi.z = min(mi.z,min(q1.z,q2.z));
      ma.z = max(ma.z,max(q1.z,q2.z));
    }
  }

  return aabb(mi - vec3(curve.r), ma + vec3(curve.r));
}

VSNRAY_FUNC inline void split_primitive(
    aabb& L, aabb& R, float plane, int axis, const BezierCurve &curve)
{
  VSNRAY_UNUSED(L);
  VSNRAY_UNUSED(R);
  VSNRAY_UNUSED(plane);
  VSNRAY_UNUSED(axis);
  VSNRAY_UNUSED(curve);

  // TODO: implement this to support SBVHs
}

// Quad (light) //
/// TODO: this geom type is at the moment only used for quad _lights_,
/// but could just as easily be used for the quad _primitive_ where
/// we currently just use two visionaray triangles

struct Quad
{
  vec3 v1, e1, e2;

  VSNRAY_FUNC
  inline void tessellate(
      basic_triangle<3,float> &t1, basic_triangle<3,float> &t2) const {
    t1.v1 = v1; t1.e1 = e1; t1.e2 = e1+e2;
    t2.v1 = v1; t2.e1 = e1+e2; t2.e2 = e2;
  }
};

VSNRAY_FUNC
inline aabb get_bounds(const Quad &q)
{
  aabb bounds;
  bounds.invalidate();
  bounds.insert(q.v1);
  bounds.insert(q.v1+q.e1);
  bounds.insert(q.v1+q.e2);
  bounds.insert(q.v1+q.e1+q.e2);
  return bounds;
}

VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersect(const Ray &ray, const Quad &q)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);

  auto hr1 = intersect(ray, t1);
  auto hr2 = intersect(ray, t2);

  hit_record<Ray, primitive<unsigned>> result;

  if (hr1.hit) {
    result.hit = true;
    result.t = hr1.t;
  }

  if (hr2.hit && hr2.t < result.t) {
    result.hit = true;
    result.t = hr2.t;
  }

  if (result.hit)
    result.isect_pos = ray.ori + ray.dir * result.t;

  return result;
}

template <typename HR>
VSNRAY_FUNC
inline vec3 get_normal(const HR &hr, const Quad &q)
{
  (void)hr;
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);
  return normalize(cross(t1.e1,t1.e2));
  //return normalize(cross(q.e1,q.e2));
}

VSNRAY_FUNC
inline float area(const Quad &q)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);
  return area(t1) + area(t2);
}

template <typename RNG>
VSNRAY_FUNC
inline vec3 sample_surface(const Quad &q, const vec3 reference_point, RNG &rng)
{
  basic_triangle<3,float> t1, t2;
  q.tessellate(t1,t2);

  float A1 = area(t1);
  float A2 = area(t2);

  float r = rng();
  if (A1/(A1+A2) < r)
    return sample_surface(t1, reference_point, rng);
  else
    return sample_surface(t2, reference_point, rng);
}

// Geometry types (for dispatch) //

using Triangle = basic_triangle<3,float>;
using Sphere = basic_sphere<float>;
using Cylinder = basic_cylinder<float>;

// Geometry //

struct Geometry
{
  enum Type {
    Triangle,
    Quad,
    Sphere,
    Cone,
    Cylinder,
    Curve,
    BezierCurve,
    ISOSurface,
    Unknown,
  };
  Type type;
  unsigned geomID;
  float surfaceArea;

  template <typename Primitive>
  VSNRAY_FUNC
  Primitive &as(unsigned primID)
  {
    return ((Primitive *)primitives.data)[primID];
  }

  template <typename Primitive>
  VSNRAY_FUNC
  const Primitive &as(unsigned primID) const
  {
    return ((const Primitive *)primitives.data)[primID];
  }

  Array primitives;
  Uniform uniformAttributes[5];
  Array primitiveAttributes[5];
  struct {
    Array attributes[5];
    Array normal;
    Array tangent;
  } vertex, faceVarying;
  Array index;
};

VSNRAY_FUNC
inline Geometry createGeometry()
{
  Geometry geom;
  memset(&geom,0,sizeof(geom));
  geom.type = Geometry::Unknown;
  geom.geomID = UINT_MAX;
  geom.surfaceArea = 0.f;
  return geom;
}

VSNRAY_FUNC
inline aabb get_bounds(const Geometry &geom)
{
  aabb result;
  result.invalidate();

  if (geom.type == dco::Geometry::Triangle) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Triangle>(i)));
    }
  } else if (geom.type == dco::Geometry::Quad) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Triangle>(i)));
    }
  } else if (geom.type == dco::Geometry::Sphere) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Sphere>(i)));
    }
  } else if (geom.type == dco::Geometry::Cone) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Cone>(i)));
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::Cylinder>(i)));
    }
  } else if (geom.type == dco::Geometry::BezierCurve) {
    for (size_t i=0;i<geom.primitives.len;++i) {
      result.insert(get_bounds(geom.as<dco::BezierCurve>(i)));
    }
  } else if (geom.type == dco::Geometry::ISOSurface) {
    if (geom.primitives.len) {
      result.insert(get_bounds(geom.as<dco::ISOSurface>(0)));
    }
  }

  return result;
}

// Surface //

struct Surface
{
  unsigned surfID;
  unsigned geomID;
  unsigned matID;
};

VSNRAY_FUNC
inline Surface createSurface()
{
  Surface surf;
  surf.surfID = UINT_MAX;
  surf.geomID = UINT_MAX;
  surf.matID = UINT_MAX;
  return surf;
}

} // namespace visionaray::dco
