#pragma once

#include <common.h>
#include <DeviceCopyableObjects.h>

namespace visionaray {

template<unsigned int N=4>
struct LCG
{
  inline VSNRAY_FUNC LCG()
  { /* intentionally empty so we can use it in device vars that
       don't allow dynamic initialization (ie, PRD) */
  }
  inline VSNRAY_FUNC LCG(unsigned int val0, unsigned int val1)
  { init(val0,val1); }

  inline VSNRAY_FUNC LCG(const vec2i &seed)
  { init((unsigned)seed.x,(unsigned)seed.y); }
  inline VSNRAY_FUNC LCG(const vec2ui &seed)
  { init(seed.x,seed.y); }
  
  inline VSNRAY_FUNC void init(unsigned int val0, unsigned int val1)
  {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
  
    for (unsigned int n = 0; n < N; n++) {
      s0 += 0x9e3779b9;
      v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
      v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    state = v0;
  }

  // Generate random unsigned int in [0, 2^24)
  inline VSNRAY_FUNC float operator() ()
  {
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    state = (LCG_A * state + LCG_C);
    return (state & 0x00FFFFFF) / (float) 0x01000000;
  }

  // For compat. with visionaray
  inline VSNRAY_FUNC float next()
  {
    return operator()();
  }

  uint32_t state;
};

typedef LCG<4> Random;

struct ScreenSample
{
  int x, y;
  uint2 frameSize;
  Random random;

  inline VSNRAY_FUNC bool debug() {
#if 1
    return x == frameSize.x/2 && y == frameSize.y/2;
#else
    return false;
#endif
  }
};

struct PixelSample
{
  float4 color;
  float depth;
};

struct RendererState
{
  float4 bgColor{float3(0.f), 1.f};
  float ambientRadiance{1.f};
  int accumID{0};
  int envID{-1};
  bool gradientShading{true};
};

inline VSNRAY_FUNC int uniformSampleOneLight(Random &rnd, int numLights)
{
  int which = int(rnd() * numLights); if (which == numLights) which = 0;
  return which;
}

VSNRAY_FUNC
inline vec3 getNormal(const dco::Geometry &geom, unsigned primID, const vec3 hitPos)
{
  vec3f gn(1.f,0.f,0.f);

        // TODO: doesn't work for instances yet
  if (geom.type == dco::Geometry::Triangle) {
    auto tri = geom.asTriangle.data[primID];
    gn = normalize(cross(tri.e1,tri.e2));
  } else if (geom.type == dco::Geometry::Sphere) {
    auto sph = geom.asSphere.data[primID];
    gn = normalize((hitPos-sph.center) / sph.radius);
  } else if (geom.type == dco::Geometry::Cylinder) {
    auto cyl = geom.asCylinder.data[primID];
    vec3f axis = normalize(cyl.v2-cyl.v1);
    if (length(hitPos-cyl.v1) < cyl.radius)
      gn = -axis;
    else if (length(hitPos-cyl.v2) < cyl.radius)
      gn = axis;
    else {
      float t = dot(hitPos-cyl.v1, axis);
      vec3f pt = cyl.v1 + t * axis;
      gn = normalize(hitPos-pt);
    }
  }
  return gn;
}

inline  VSNRAY_FUNC vec4f over(const vec4f &A, const vec4f &B)
{
  return A + (1.f-A.w)*B;
}

VSNRAY_FUNC
inline void print(const float3 &v)
{
  printf("float3: (%f,%f,%f)\n", v.x, v.y, v.z);
}

VSNRAY_FUNC
inline void print(const aabb &box)
{
  printf("aabb: [min: (%f,%f,%f), max: (%f,%f,%f)]\n",
      box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z);
}

VSNRAY_FUNC
inline void print(const Ray &ray)
{
  printf("ray: [ori: (%f,%f,%f), dir: (%f,%f,%f), tmin: %f, %f, mask: %u]\n",
      ray.ori.x, ray.ori.y, ray.ori.z, ray.dir.x, ray.dir.y, ray.dir.z,
      ray.tmin, ray.tmax, ray.intersectionMask);
}

} // visionaray
