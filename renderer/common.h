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

enum class RenderMode
{
  Default,
  Ng,
  GeometryAttribute0,
  GeometryAttribute1,
  GeometryAttribute2,
  GeometryAttribute3,
  GeometryColor,
};

struct RendererState
{
  float4 bgColor{float3(0.f), 1.f};
  float ambientRadiance{1.f};
  RenderMode renderMode{RenderMode::Default};
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

VSNRAY_FUNC
inline dco::Array getVertexColors(const dco::Geometry &geom, dco::Attribute attrib)
{
  dco::Array arr;

  if (attrib != dco::Attribute::None) {
    if (geom.type == dco::Geometry::Triangle)
      return geom.asTriangle.vertexAttributes[(int)attrib];
    else if (geom.type == dco::Geometry::Sphere)
      return geom.asSphere.vertexAttributes[(int)attrib];
  }

  return arr;
}

VSNRAY_FUNC
inline dco::Array getPrimitiveColors(const dco::Geometry &geom, dco::Attribute attrib)
{
  dco::Array arr;

  if (attrib != dco::Attribute::None)
    return geom.primitiveAttributes[(int)attrib];

  return arr;
}

VSNRAY_FUNC
inline vec4 getAttribute(
    const dco::Geometry &geom, dco::Attribute attrib, unsigned primID, const vec2 uv,
    const vec4 dflt = vec4(0.f))
{
  vec4f color = dflt;
  dco::Array vertexColors = getVertexColors(geom, attrib);
  dco::Array primitiveColors = getPrimitiveColors(geom, attrib);

  // vertex colors take precedence over primitive colors
  if (geom.type == dco::Geometry::Triangle && vertexColors.len > 0) {
    if (vertexColors.type == ANARI_FLOAT32_VEC4) {
      vec4f c1, c2, c3;
      if (geom.asTriangle.index.len > 0) {
        uint3 index = ((uint3 *)geom.asTriangle.index.data)[primID];
        c1 = ((vec4f *)vertexColors.data)[index.x];
        c2 = ((vec4f *)vertexColors.data)[index.y];
        c3 = ((vec4f *)vertexColors.data)[index.z];
      } else {
        c1 = ((vec4f *)vertexColors.data)[primID * 3];
        c2 = ((vec4f *)vertexColors.data)[primID * 3 + 1];
        c3 = ((vec4f *)vertexColors.data)[primID * 3 + 2];
      }
      color = lerp(c1, c2, c3, uv.x, uv.y);
    }
  }
  else if (geom.type == dco::Geometry::Sphere && vertexColors.len > 0) {
    if (vertexColors.type == ANARI_FLOAT32) {
      float val = 0.f;
      if (geom.asSphere.index.len > 0) {
        uint32_t index = ((uint32_t *)geom.asSphere.index.data)[primID];
        val = ((float *)vertexColors.data)[index];
      } else {
        val = ((float *)vertexColors.data)[primID];
      }
      color = {val,0.f,0.f,0.f};
    } else if (vertexColors.type == ANARI_FLOAT32_VEC4) {
      if (geom.asSphere.index.len > 0) {
        uint32_t index = ((uint32_t *)geom.asSphere.index.data)[primID];
        color = ((vec4f *)vertexColors.data)[index];
      } else {
        color = ((vec4f *)vertexColors.data)[primID];
      }
    }
  }
  else if (primitiveColors.len > 0) {
    if (primitiveColors.type == ANARI_FLOAT32_VEC4) {
      color = ((vec4f *)primitiveColors.data)[primID];
    }
  }

  return color;
}

VSNRAY_FUNC
inline vec4 getSample(
    const dco::Sampler &samp, const dco::Geometry geom, unsigned primID, const vec2 uv)
{
  vec4f inAttr = getAttribute(geom, samp.inAttribute, primID, uv);

  if (samp.type == dco::Sampler::Image1D)
    return tex1D(samp.asImage1D, inAttr.x);
  else if (samp.type == dco::Sampler::Image2D)
    return tex2D(samp.asImage2D, inAttr.xy());

  return vec4{0.f};
}

VSNRAY_FUNC
inline vec4 getColor(
    const dco::Geometry &geom, const dco::Material &mat, unsigned primID, const vec2 uv)
{
  vec4f defaultColor(1.f);
  if (mat.type == dco::Material::Matte) {
    if (mat.asMatte.colorSampler.isValid()) {
      defaultColor = getSample(mat.asMatte.colorSampler, geom, primID, uv);
    } else {
      defaultColor = vec4f(to_rgb(mat.asMatte.data.cd()), 1.f);
    }
  }
  return getAttribute(geom, mat.colorAttribute, primID, uv, defaultColor);
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
