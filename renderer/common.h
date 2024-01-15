#pragma once

// visionaray
#include "visionaray/material.h"
// ours
#include "common.h"
#include "DeviceCopyableObjects.h"
#include "VisionarayGlobalState.h"

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
  int frameID;
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

enum class RenderMode
{
  Default,
  Ng,
  Albedo,
  MotionVec,
  GeometryAttribute0,
  GeometryAttribute1,
  GeometryAttribute2,
  GeometryAttribute3,
  GeometryColor,
};

struct RendererState
{
  float4 bgColor{float3(0.f), 1.f};
  RenderMode renderMode{RenderMode::Default};
  int pixelSamples{1};
  int accumID{0};
  int envID{-1};
  // TAA
  bool taaEnabled{false};
  float taaAlpha{0.3f};
  mat4 prevMV{mat4::identity()};
  mat4 prevPR{mat4::identity()};
  mat4 currMV{mat4::identity()};
  mat4 currPR{mat4::identity()};
  // Volume
  bool gradientShading{false};
  // AO
  float3 ambientColor{1.f, 1.f, 1.f};
  float ambientRadiance{0.2f};
  float occlusionDistance{1e20f};
  int ambientSamples{1};
  // Heat map
  bool heatMapEnabled{false};
  float heatMapScale{.1f};

};

inline VSNRAY_FUNC
vec3 hsv2rgb(vec3 in)
{
    float      hh, p, q, t, ff;
    long        i;
    vec3         out;

    if(in.y <= 0.0) {       // < is bogus, just shuts up warnings
        out.x = in.z;
        out.y = in.z;
        out.z = in.z;
        return out;
    }
    hh = in.x;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.z * (1.0 - in.y);
    q = in.z * (1.0 - (in.y * ff));
    t = in.z * (1.0 - (in.y * (1.0 - ff)));

    switch(i) {
        case 0:
            out.x = in.z;
            out.y = t;
            out.z = p;
            break;
        case 1:
            out.x = q;
            out.y = in.z;
            out.z = p;
            break;
        case 2:
            out.x = p;
            out.y = in.z;
            out.z = t;
            break;

        case 3:
            out.x = p;
            out.y = q;
            out.z = in.z;
            break;
        case 4:
            out.x = t;
            out.y = p;
            out.z = in.z;
            break;
        case 5:
        default:
            out.x = in.z;
            out.y = p;
            out.z = q;
            break;
    }
    return out;
}

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
  } else if (geom.type == dco::Geometry::ISOSurface) {
    if (!sampleGradient(geom.asISOSurface.data.field,hitPos,gn)) {
      return vec3f(0.f);
    }
    gn = normalize(gn);
  }
  return gn;
}

VSNRAY_FUNC
inline vec3 getShadingNormal(
    const dco::Geometry &geom, unsigned primID, const vec3 hitPos, const vec2 uv)
{
  vec3f sn(1.f,0.f,0.f);

  if (geom.type == dco::Geometry::Triangle) {
    if (geom.asTriangle.normal.len
        && geom.asTriangle.normal.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
      uint3 index;
      if (geom.asTriangle.index.len > 0) {
        index = ((uint3 *)geom.asTriangle.index.data)[primID];
      } else {
        index = uint3(primID * 3, primID * 3 + 1, primID * 3 + 2);
      }
      auto *normals = (const vec3 *)geom.asTriangle.normal.data;
      vec3 n1 = normals[index.x];
      vec3 n2 = normals[index.y];
      vec3 n3 = normals[index.z];
      sn = lerp(n1, n2, n3, uv.x, uv.y);
    } else {
      sn = getNormal(geom, primID, hitPos);
    }
  } else {
    sn = getNormal(geom, primID, hitPos);
  }

  return sn;
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
    else if (geom.type == dco::Geometry::Cylinder)
      return geom.asCylinder.vertexAttributes[(int)attrib];
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
    const dco::Geometry &geom, dco::Attribute attrib, unsigned primID, const vec2 uv)
{
  const vec4 dflt{0.f, 0.f, 0.f, 1.f};
  vec4f color = dflt;
  dco::Array vertexColors = getVertexColors(geom, attrib);
  dco::Array primitiveColors = getPrimitiveColors(geom, attrib);

  const TypeInfo &vertexColorInfo = vertexColors.typeInfo;
  const TypeInfo &primitiveColorInfo = primitiveColors.typeInfo;

  // vertex colors take precedence over primitive colors
  if (geom.type == dco::Geometry::Triangle && vertexColors.len > 0) {
    uint3 index;
    if (geom.asTriangle.index.len > 0) {
      index = ((uint3 *)geom.asTriangle.index.data)[primID];
    } else {
      index = uint3(primID * 3, primID * 3 + 1, primID * 3 + 2);
    }
    const auto *source1
        = (const uint8_t *)vertexColors.data
            + index.x * vertexColorInfo.sizeInBytes;
    const auto *source2
        = (const uint8_t *)vertexColors.data
            + index.y * vertexColorInfo.sizeInBytes;
    const auto *source3
        = (const uint8_t *)vertexColors.data
            + index.z * vertexColorInfo.sizeInBytes;
    vec4f c1{dflt}, c2{dflt}, c3{dflt};
    convert(&c1, source1, vertexColorInfo);
    convert(&c2, source2, vertexColorInfo);
    convert(&c3, source3, vertexColorInfo);
    color = lerp(c1, c2, c3, uv.x, uv.y);
  }
  else if (geom.type == dco::Geometry::Sphere && vertexColors.len > 0) {
    uint32_t index;
    if (geom.asSphere.index.len > 0) {
      index = ((uint32_t *)geom.asSphere.index.data)[primID];
    } else {
      index = primID;
    }
    const auto *source
        = (const uint8_t *)vertexColors.data
            + index * vertexColorInfo.sizeInBytes;
    convert(&color, source, vertexColorInfo);
  }
  else if (geom.type == dco::Geometry::Cylinder && vertexColors.len > 0) {
    uint2 index;
    if (geom.asCylinder.index.len > 0) {
      index = ((uint2 *)geom.asCylinder.index.data)[primID];
    } else {
      index = uint2(primID * 2, primID * 2 + 1);
    }
    const auto *source1
        = (const uint8_t *)vertexColors.data
            + index.x * vertexColorInfo.sizeInBytes;
    const auto *source2
        = (const uint8_t *)vertexColors.data
            + index.y * vertexColorInfo.sizeInBytes;
    vec4f c1{dflt}, c2{dflt};
    convert(&c1, source1, vertexColorInfo);
    convert(&c2, source2, vertexColorInfo);
    color = lerp(c1, c2, uv.x);
  }
  else if (primitiveColors.len > 0) {
    const auto *source
        = (const uint8_t *)primitiveColors.data
            + primID * primitiveColorInfo.sizeInBytes;
    convert(&color, source, primitiveColorInfo);
  }

  return color;
}

VSNRAY_FUNC
inline vec4 getSample(
    const dco::Sampler &samp, const dco::Geometry geom, unsigned primID, const vec2 uv)
{
  vec4f s{0.f, 0.f, 0.f, 1.f};

  if (samp.type == dco::Sampler::Primitive) {
    const TypeInfo &info = samp.asPrimitive.typeInfo;
    const auto *source = samp.asPrimitive.data
        + (primID * info.sizeInBytes) + (samp.asPrimitive.offset * info.sizeInBytes);
    convert(&s, source, info);
  } else if (samp.type == dco::Sampler::Transform) {
    vec4f inAttr = getAttribute(geom, samp.inAttribute, primID, uv);
    s = samp.outTransform * inAttr + samp.outOffset;
  } else {
    vec4f inAttr = getAttribute(geom, samp.inAttribute, primID, uv);

    inAttr = samp.inTransform * inAttr + samp.inOffset;

    if (samp.type == dco::Sampler::Image1D)
      s = tex1D(samp.asImage1D, inAttr.x);
    else if (samp.type == dco::Sampler::Image2D)
      s = tex2D(samp.asImage2D, inAttr.xy());
    else if (samp.type == dco::Sampler::Image3D)
      s = tex3D(samp.asImage3D, inAttr.xyz());

    s = samp.outTransform * s + samp.outOffset;
  }

  return s;
}

VSNRAY_FUNC
inline vec4 getRGBA(const dco::MaterialParamRGB &param,
                    const dco::Geometry &geom,
                    const dco::Sampler *samplers,
                    unsigned primID, const vec2 uv)
{
  vec4f rgba{0.f, 0.f, 0.f, 1.f};
  if (param.samplerID < UINT_MAX) {
    const auto &samp = samplers[param.samplerID];
    rgba = getSample(samp, geom, primID, uv);
  } else if (param.attribute != dco::Attribute::None) {
    rgba = getAttribute(geom, param.attribute, primID, uv);
  } else {
    rgba = vec4f(param.rgb, 1.f);
  }
  return rgba;
}

VSNRAY_FUNC
inline float getF(const dco::MaterialParamF &param,
                  const dco::Geometry &geom,
                  const dco::Sampler *samplers,
                  unsigned primID, const vec2 uv)
{
  float f = 1.f;
  if (param.samplerID < UINT_MAX) {
    const auto &samp = samplers[param.samplerID];
    f = getSample(samp, geom, primID, uv).x;
  } else if (param.attribute != dco::Attribute::None) {
    f = getAttribute(geom, param.attribute, primID, uv).x;
  } else {
    f = param.f;
  }
  return f;
}

VSNRAY_FUNC
inline vec4 getColor(const dco::Material &mat,
                     const dco::Geometry &geom,
                     const dco::Sampler *samplers,
                     unsigned primID, const vec2 uv)
{
  vec4f color{0.f, 0.f, 0.f, 1.f};
  if (mat.type == dco::Material::Matte)
    color = getRGBA(mat.asMatte.color, geom, samplers, primID, uv);
  else if (mat.type == dco::Material::PhysicallyBased) {
    const float metallic = getF(
        mat.asPhysicallyBased.metallic, geom, samplers, primID, uv);
    color = getRGBA(mat.asPhysicallyBased.baseColor, geom, samplers, primID, uv);
    color = lerp(color, vec4f(0.f, 0.f, 0.f, 1.f), metallic);
  }
  return color;
}

VSNRAY_FUNC
inline float getOpacity(const dco::Material &mat,
                        const dco::Geometry &geom,
                        const dco::Sampler *samplers,
                        unsigned primID, const vec2 uv)
{
  float opacity = 1.f;
  if (mat.type == dco::Material::Matte)
    opacity = getF(mat.asMatte.opacity, geom, samplers, primID, uv);
  else if (mat.type == dco::Material::PhysicallyBased) {
    opacity = getF(mat.asPhysicallyBased.opacity, geom, samplers, primID, uv);
  }
  return opacity;
}

VSNRAY_FUNC
inline vec3 evalPhysicallyBasedMaterial(const dco::Material &mat,
                                        const dco::Geometry &geom,
                                        const dco::Sampler *samplers,
                                        unsigned primID, const vec2 uv,
                                        const vec3 Ng, const vec3 Ns,
                                        const vec3 viewDir, const vec3 lightDir,
                                        const vec3 lightIntensity)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, geom, samplers, primID, uv);
  const float roughness = getF(
      mat.asPhysicallyBased.roughness, geom, samplers, primID, uv);
  const float ior = getF(
      mat.asPhysicallyBased.ior, geom, samplers, primID, uv);

  const float alpha = roughness*roughness;

  const vec3 H = normalize(lightDir+viewDir);
  const float NdotH = dot(Ns,H);
  const float NdotL = dot(Ns,lightDir);
  const float NdotV = dot(Ns,viewDir);
  const float VdotH = dot(viewDir,H);
  const float LdotH = dot(lightDir,H);

  // Diffuse:
  vec3 diffuseColor = getRGBA(
      mat.asPhysicallyBased.baseColor, geom, samplers, primID, uv).xyz();

  // Fresnel
  auto pow2 = [](float f) { return f*f; };
  auto pow5 = [](float f) { return f*f*f*f*f; };
  vec3 f0 = lerp(vec3(pow2((1.f-ior)/(1.f+ior))), diffuseColor, metallic);
  vec3 F = f0 + (vec3(1.f) - f0) * pow5(1.f - fabsf(VdotH));

  // Metallic materials don't reflect diffusely:
  diffuseColor = lerp(diffuseColor, vec3f(0.f), metallic);

  vec3 diffuseBRDF
      = (vec3(1.f)-F) * constants::inv_pi<float>() * diffuseColor * fmaxf(0.f,NdotL);

  // GGX microfacet distribution
  float D = (alpha*alpha*heaviside(NdotH))
    / (constants::pi<float>()*pow2(NdotH*NdotH*(alpha*alpha-1.f)+1.f));

  // Masking-shadowing term
  float G = ((2.f * fabsf(NdotL) * heaviside(LdotH))
        / (fabsf(NdotL) + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotL*NdotL)))
    *       ((2.f * fabsf(NdotV) * heaviside(VdotH))
        / (fabsf(NdotV) + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotV*NdotV)));

  float denom = 4.f * fabsf(NdotV) * fabsf(NdotL);
  vec3 specularBRDF = denom != 0.f ? (F * D * G) / denom : vec3(0.f);

  return (diffuseBRDF + specularBRDF) * lightIntensity;
}

VSNRAY_FUNC
inline vec3 evalMaterial(const dco::Material &mat,
                         const dco::Geometry &geom,
                         const dco::Sampler *samplers,
                         unsigned primID, const vec2 uv,
                         const vec3 Ng, const vec3 Ns,
                         const vec3 viewDir, const vec3 lightDir,
                         const vec3 lightIntensity)
{
  vec3 shadedColor{0.f, 0.f, 0.f};
  if (mat.type == dco::Material::Matte) {
    vec4f color = getColor(mat, geom, samplers, primID, uv);

    shade_record<float> sr;
    sr.normal = Ns;
    sr.geometric_normal = Ng;
    sr.view_dir = viewDir;
    sr.tex_color = float3(1.f);
    sr.light_dir = normalize(lightDir);
    sr.light_intensity = lightIntensity;

    matte<float> vmat;
    vmat.cd() = from_rgb(color.xyz());
    vmat.kd() = 1.f;

    shadedColor = to_rgb(vmat.shade(sr));
  } else if (mat.type == dco::Material::PhysicallyBased) {
    shadedColor = evalPhysicallyBasedMaterial(mat,
                                              geom,
                                              samplers,
                                              primID, uv,
                                              Ng, Ns,
                                              viewDir, lightDir,
                                              lightIntensity);
  }
  return shadedColor;
}

template <bool EvalOpacity>
VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    ScreenSample &ss, Ray ray,
    const VisionarayGlobalState::DeviceObjectRegistry &onDevice,
    unsigned worldID)
{
  auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
  while (EvalOpacity) {
    if (!hr.hit) break;

    float2 uv = toUV(ray.dir);
    const dco::Instance &inst = onDevice.instances[hr.inst_id];
    const dco::Group &group = onDevice.groups[inst.groupID];
    const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
    const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];
    float opacity = getOpacity(mat, geom, onDevice.samplers, hr.prim_id, uv);

    float r = ss.random();
    if (r > opacity) {
      ray.tmin = hr.t + 1e-4f;
      hr = intersectSurfaces(ray, onDevice.TLSs[worldID]);
    } else {
      break;
    }
  }
  return hr;
}

inline  VSNRAY_FUNC vec4f over(const vec4f &A, const vec4f &B)
{
  return A + (1.f-A.w)*B;
}

inline VSNRAY_FUNC vec3f hue_to_rgb(float hue)
{
  float s = saturate( hue ) * 6.0f;
  float r = saturate( fabsf(s - 3.f) - 1.0f );
  float g = saturate( 2.0f - fabsf(s - 2.0f) );
  float b = saturate( 2.0f - fabsf(s - 4.0f) );
  return vec3f(r, g, b); 
}
  
inline VSNRAY_FUNC vec3f temperature_to_rgb(float t)
{
  float K = 4.0f / 6.0f;
  float h = K - K * t;
  float v = .5f + 0.5f * t;    return v * hue_to_rgb(h);
}
  
                                  
inline VSNRAY_FUNC
vec3f heatMap(float t)
{
#if 1
  return temperature_to_rgb(t);
#else
  if (t < .25f) return lerp(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
  if (t < .5f)  return lerp(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
  if (t < .75f) return lerp(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
  if (t < 1.f)  return lerp(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
  return vec3f(1.f,0.f,0.f);
#endif
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
