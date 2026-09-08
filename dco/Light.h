// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/area_light.h"
#include "visionaray/directional_light.h"
#include "visionaray/sampling.h"
// ours
#include "dco/common.h"
#include "dco/sampleCDF.h"

namespace visionaray::dco {

// Light //

struct Light
{
  enum Type { Directional, Point, Quad, Spot, HDRI, Geometry, Unknown, };
  enum Side { Front, Back, Both, };

  Type type;
  unsigned lightID;
  bool visible;

  VSNRAY_FUNC
  inline bool isAreaLight() const {
    return type == Quad ||
           type == HDRI ||
           type == Geometry ||
           (type == Directional && asDirectional.angular_diameter() > 0.f) ||
           (type == Point && asPoint.radius > 0.f);
  }

  VSNRAY_FUNC
  inline float3 radiance(float3 pos, float3 dir) const {
    if (type == Directional)
      return asDirectional.intensity(pos);
    else if (type == Point)
      return asPoint.radiance(pos);
    else if (type == Quad)
      return asQuad.radiance(dir);
    else if (type == Spot)
      return asSpot.intensity(dir);
    else if (type == HDRI)
      return asHDRI.radiance(dir);
    // else if (type == Geometry)
    //   return asGeometry.radiance(dir);
    // ^^^ geometry lights have no radiance() function ^^^
    // instead, we hit the geometry associated with them w/
    // our ray and then directly evaluate the emissive component

    return {};
  }

  union {
    directional_light<float> asDirectional;
    // point light:
    struct {
      float3 position;
      float3 color;
      float lightIntensity;
      float radius;

      template <typename RNG>
      VSNRAY_FUNC
      inline light_sample<float> sample(const float3 &refPoint, RNG &rng) const
      {
        light_sample<float> result;
        if (radius < FLT_MIN) {
          result.dir = position-refPoint;
          result.dist = length(result.dir);
          result.normal = normalize(
              float3(rng() * 2.f - 1.f, rng() * 2.f - 1.f, rng() * 2.f - 1.f));
          result.area = 1.f;
          result.delta_light = true;
          result.pdf = 1.f;
        } else {
          float3 centerDir = position-refPoint;
          float d2 = norm2(centerDir);
          float r2 = radius*radius;

          if (d2 <= r2) {
            // case that ref-point is inside the sphere or on the boundary
            float3 Nl = uniform_sample_sphere(rng(),rng()); // sampled unit dir / normal
            float3 pos = Nl * radius + position;
            result.dir = pos-refPoint;
            result.dist = length(result.dir);
            result.pdf = 1.f / (4.f*constants::pi<float>());
          } else {
            float3 u, v;
            float3 w = normalize(centerDir);
            make_orthonormal_basis(u, v, w);

            float sintSqMax = fminf(r2/d2,1.f);
            float costMax = sqrtf(fmaxf(0.f, 1.f-sintSqMax));
            float oneMinusCostMax = sintSqMax / (1.f+costMax);

            float u1 = rng(), u2 = rng();

            // sample direction within subtended cone:
            float oneMinusCost = u1 * oneMinusCostMax;
            float cost = 1.f-oneMinusCost;
            float sint = sqrtf(fmaxf(0.f, oneMinusCost * (1.f+cost)));
            float phi  = constants::two_pi<float>() * u2;

            float3 localDir(cosf(phi) * sint, sinf(phi) * sint, cost);

            float dist = length(centerDir) * cost - sqrtf(fmaxf(0.f,r2-d2*(1-cost*cost)));
            result.dir = (localDir.x*u + localDir.y*v + localDir.z*w) * dist;
            result.dist = dist;
            float solidAngle = constants::two_pi<float>() * oneMinusCostMax;
            result.pdf = (solidAngle > 1e-12f) ? (1.f / solidAngle) : 0.f;
          }

          result.normal = normalize(-result.dir);
          result.area = 4.f*constants::pi<float>()*radius*radius;
          result.delta_light = false;
        }
        return result;
      }

      VSNRAY_FUNC
      inline float3 radiance(const float3 &refPoint) const
      {
        if (radius < FLT_MIN) {
          float ld2 = norm2(refPoint-position);
          return (color * lightIntensity) / ld2;
        }

        float disc_area = constants::pi<float>() * radius * radius;
        return (color * lightIntensity) / disc_area;
      }

      VSNRAY_FUNC
      inline float pdf(const Ray &ray, const float3 hitPos) const
      {
        if (radius < FLT_MIN)
          return 0.f;

        float d2 = norm2(ray.ori - hitPos);
        float r2 = radius * radius;

        if (d2 <= r2) return 1.f / (4.f * constants::pi<float>());

        float sintSqMax = fminf(r2/d2, 1.f);
        float costMax = sqrtf(fmaxf(0.f, 1.f - sintSqMax));

        float oneMinusCostMax = sintSqMax / (1.f + costMax);
        float solidAngle = constants::two_pi<float>() * oneMinusCostMax;
        return (solidAngle > 1e-12f) ? (1.f / solidAngle) : 0.f;
      }
    } asPoint;
    // spot light:
    struct {
      float3 position;
      float3 direction;
      float cosOuterAngle;
      float cosInnerAngle;
      float3 color;
      float lightIntensity;

      template <typename RNG>
      VSNRAY_FUNC
      inline light_sample<float> sample(const float3 &refPoint, RNG &rng) const
      {
        light_sample<float> result;
        result.dir = position-refPoint;
        result.dist = length(result.dir);
        result.normal = normalize(
            float3(rng() * 2.f - 1.f, rng() * 2.f - 1.f, rng() * 2.f - 1.f));
        result.area = 1.f;
        result.delta_light = true;
        result.pdf = 1.f;
        return result;
      }

      VSNRAY_FUNC
      inline float3 intensity(const float3 lightDir) const
      {
        // compute intensity
        float spot = dot(normalize(direction), normalize(-lightDir));
        if (spot < cosOuterAngle) return float3(0.f);
        if (spot > cosInnerAngle) return color * lightIntensity;
        spot = (spot - cosOuterAngle) / (cosInnerAngle - cosOuterAngle);
        spot = spot * spot * (3.f - 2.f * spot);
        return color * lightIntensity * spot;
      }
    } asSpot;
    // quad light:
    struct {
      area_light<float,dco::Quad> internal;
      Side side;

      VSNRAY_FUNC
      inline dco::Quad &geometry()
      { return internal.geometry(); }

      VSNRAY_FUNC
      inline const dco::Quad &geometry() const
      { return internal.geometry(); }

      VSNRAY_FUNC
      inline void set_cl(const float3 &cl)
      { internal.set_cl(cl); }

      VSNRAY_FUNC
      inline void set_kl(float kl)
      { internal.set_kl(kl); }

      template<typename RNG>
      VSNRAY_FUNC
      inline light_sample<float> sample(const float3 &refPoint, RNG &rng) const
      {
        // Quad light sampling technique by Urena et al. (2013)
        // An Area-Preserving Parametrization for Spherical Rectangles

        light_sample<float> ls{};

        struct {
          float3 o, x, y, z;      // local reference system 'R'
          float z0, z0sq;
          float x0, y0, y0sq;        //
          float x1, y1, y1sq;        // rectangle coords in 'R'
          float x2, y2, y2sq;        //
          float b0, b1, b0sq, k;  // misc precomputed constants
          float S;                // solid angle of 'Q'
        } squad;

        // --- init squad -------------
        {
        squad.o = refPoint;
        float exl = length(geometry().e1), eyl = length(geometry().e2);
        // compute local reference system 'R'
        squad.x = geometry().e1 / exl;
        squad.y = geometry().e2 / eyl;
        squad.z = cross(squad.x, squad.y);
        // compute rectangle coords in local reference system
        float3 d = geometry().v1 - refPoint;
        squad.z0 = dot(d, squad.z);
        // flip 'z' to make it point against 'Q'
        if (squad.z0 > 0) {
          squad.z  *= -1.f;
          squad.z0 *= -1.f;
        }
        squad.z0sq = squad.z0 * squad.z0;
        squad.x0 = dot(d, squad.x);
        squad.y0 = dot(d, squad.y);
        squad.x1 = squad.x0 + exl;
        squad.y1 = squad.y0 + eyl;
        squad.y0sq = squad.y0 * squad.y0;
        squad.y1sq = squad.y1 * squad.y1;
        // create vectors to four vertices
        float3 v00(squad.x0, squad.y0, squad.z0);
        float3 v01(squad.x0, squad.y1, squad.z0);
        float3 v10(squad.x1, squad.y0, squad.z0);
        float3 v11(squad.x1, squad.y1, squad.z0);
        // compute normals to edges
        float3 n0 = normalize(cross(v00, v10));
        float3 n1 = normalize(cross(v10, v11));
        float3 n2 = normalize(cross(v11, v01));
        float3 n3 = normalize(cross(v01, v00));
        // compute internal angles (gamma_i)
        float g0 = acosf(-dot(n0,n1));
        float g1 = acosf(-dot(n1,n2));
        float g2 = acosf(-dot(n2,n3));
        float g3 = acosf(-dot(n3,n0));
        // compute predefined constants
        squad.b0 = n0.z;
        squad.b1 = n2.z;
        squad.b0sq = squad.b0 * squad.b0;
        squad.k = constants::two_pi<float>() - g2 - g3;
        // compute solid angle from internal angles
        squad.S = g0 + g1 - squad.k;
        }

        if (squad.S == 0.f) {
          // projected area is 0
          ls.pdf = 0.f;
          return ls;
        }

        // --- sampling ---------------

        float u = rng(), v = rng();
        float eps = 1e-10f;

        // 1. compute 'cu'
        float au = u * squad.S + squad.k;
        float fu = (cosf(au) * squad.b0 - squad.b1) / sinf(au);
        float cu = 1/sqrtf(fu*fu + squad.b0sq) * (fu>0 ? +1 : -1);
              cu = clamp(cu, -1.f, 1.f); // avoid NaNs
        // 2. compute 'xu'
        float xu = -(cu * squad.z0) / sqrtf(1 - cu*cu);
              xu = clamp(xu, squad.x0, squad.x1); // avoid Infs
        // 3. compute 'yv'
        float d  = sqrtf(xu*xu + squad.z0sq);
        float h0 = squad.y0 / sqrtf(d*d + squad.y0sq);
        float h1 = squad.y1 / sqrtf(d*d + squad.y1sq);
        float hv = h0 + v * (h1-h0), hv2 = hv*hv;
        float yv = (hv2 < 1-eps) ? (hv*d)/sqrtf(1-hv2) : squad.y1;
        // 4. transform (xu,yv,z0) to world coors
        float3 p(squad.o + xu*squad.x + yv*squad.y + squad.z0*squad.z);

        // Satisfy get_normal() interface
        struct { float3 isect_pos; } hr;
        hr.isect_pos = p;

        ls.dir = p - refPoint;
        ls.dist = length(ls.dir);
        ls.intensity = radiance(ls.dir);
        ls.normal = get_normal(hr, geometry());
        ls.area = area(geometry());
        ls.delta_light = false;
        ls.pdf = 1.f/squad.S;

        return ls;
      }

      VSNRAY_FUNC
      inline float3 radiance(const float3 &lightDir) const
      { return internal.intensity(lightDir) / area(geometry()); }

      VSNRAY_FUNC
      inline float pdf(const Ray &ray, const float3 hitPos) const
      {
        float A = area(geometry());
        float ld = length(hitPos-ray.ori);
        float3 L = normalize(hitPos-ray.ori);
        float3 Nl = get_normal(hit_record<Ray,primitive<unsigned>>{},geometry());
        float LdotNl = fmaxf(1e-10f,fabsf(dot(-L,Nl)));
        //float solidAngle = (LdotNl*A) / (ld*ld);
        //return 1.f/solidAngle;
        return (ld*ld) / (LdotNl*A);
      }

    } asQuad;
    // HDRI:
    struct {
#ifdef WITH_CUDA
      cuda_texture_ref<float4, 2> radianceTexture;
#elif defined(WITH_HIP)
      hip_texture_ref<float4, 2> radianceTexture;
#else
      texture_ref<float4, 2> radianceTexture;
#endif
      float scale;
      mat3 toWorld;
      mat3 toLocal;
      struct CDF {
        float *rows;
        float *lastCol;
        unsigned width;
        unsigned height;
      } cdf;

      template <typename RNG>
      VSNRAY_FUNC
      inline light_sample<float> sample(const float3 &refPoint, RNG &rng) const
      {
        CDFSample sample = sampleCDF(cdf.rows, cdf.lastCol, cdf.width, cdf.height, rng(), rng());
        float invjacobian = cdf.width*cdf.height/float(4*M_PI);
        float3 L(toPolar(float2(sample.x/float(cdf.width), sample.y/float(cdf.height))));
        light_sample<float> ls;
        ls.dir = toWorld*L;
        ls.normal = -ls.dir;
        ls.dist = FLT_MAX;
        ls.pdf = sample.pdfx*sample.pdfy*invjacobian;
        return ls;
      }

      VSNRAY_FUNC
      inline float3 radiance(const float3 dir) const
      {
        return tex2D(radianceTexture, toUV(toLocal*dir)).xyz()*scale;
      }

      VSNRAY_FUNC
      inline float pdf(const Ray &ray, const float3 hitPos) const
      {
        float3 dir = toLocal*ray.dir;
        float2 uv = toUV(dir);
        CDFSample sample = sampleCDF(cdf.rows, cdf.lastCol,
                                     cdf.width, cdf.height,
                                     uv.x, uv.y);
        float theta = acosf(clamp(dir.y, -1.0f, 1.0f));
        float sinTheta = sinf(theta);
        if (sinTheta != 0.f) {
          return (sample.pdfx * sample.pdfy) * (cdf.width * cdf.height)
              / (2.0f * constants::pi<float>() * constants::pi<float>() * sinTheta);
        } else {
          return 0.f;
        }
      }

    } asHDRI;
    struct {
      unsigned geomID;
      unsigned matID;
      float surfaceArea;

    } asGeometry;
  };
};

VSNRAY_FUNC
inline Light xfmLight(const Light &light, const mat4 &xfm)
{
  Light result = light;
  if (light.type == Light::Point) {
    float4 pos(light.asPoint.position,1.f);
    pos = xfm * pos;
    result.asPoint.position = pos.xyz();
  } else if (light.type == Light::Directional) {
    float3 dir = light.asDirectional.direction();
    mat3 LU = top_left(xfm);
    result.asDirectional.set_direction(LU * dir);
  } else if (light.type == Light::Spot) {
    float4 pos(light.asSpot.position, 1.f);
    float3 dir = light.asSpot.direction;
    pos = xfm * pos;
    mat3 LU = top_left(xfm);
    result.asSpot.position = pos.xyz();
    result.asSpot.direction = LU * dir;
  } else if (light.type == Light::Quad) {
    float4 v1(light.asQuad.geometry().v1, 1.f);
    float3 e1 = light.asQuad.geometry().e1;
    float3 e2 = light.asQuad.geometry().e2;
    v1 = xfm * v1;
    mat3 LU = top_left(xfm);
    e1 = LU * e1;
    e2 = LU * e2;
    result.asQuad.geometry().v1 = v1.xyz();
    result.asQuad.geometry().e1 = e1;
    result.asQuad.geometry().e2 = e2;
  } else {
    // TODO!
  }
  return result;
}

VSNRAY_FUNC
inline Light createLight()
{
  Light light;
  memset(&light,0,sizeof(light));
  light.type = Light::Unknown;
  light.lightID = UINT_MAX;
  light.visible = true;
  return light;
}

// LightRef associates a light with an instance
struct LightRef
{
  unsigned lightID;
  unsigned instID;
};

// Light source samplers //

// Simple uniform sampler
struct UniformLightSampler
{
  struct Sample { unsigned lightID; float pdf; };

  VSNRAY_FUNC
  inline Sample sample(Random &rnd)
  {
    if (numLights == 0)
      return { UINT_MAX, 0.f };

    unsigned which = unsigned(rnd() * numLights); if (which == numLights) which = 0;
    return { which, 1.f/numLights };
  }

  unsigned numLights;
};


// 'polymorphic', so we can switch at runtime (TODO: do we need that?)
struct LightSampler
{
  enum Type { Uniform, Unknown, } type;
  struct Sample { unsigned lightID; float pdf; };

  VSNRAY_FUNC
  inline unsigned numLights() const
  {
    if (type == Uniform) {
      return asUniform.numLights;
    }

    return 0u;
  }

  VSNRAY_FUNC
  inline Sample sample(Random &rnd)
  {
    if (type == Uniform) {
      auto s = asUniform.sample(rnd);
      return {s.lightID,s.pdf};
    }

    return { UINT_MAX, 0.f };
  }

  union {
    UniformLightSampler asUniform;
  };
};

VSNRAY_FUNC
inline LightSampler createLightSampler()
{
  LightSampler lightSampler;
  lightSampler.type = LightSampler::Unknown;
  return lightSampler;
}

} // namespace visionaray::dco
