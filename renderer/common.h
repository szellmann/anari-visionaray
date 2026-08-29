// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ours
#include "common.h"
#include "DeviceCopyableObjects.h"
#include "DeviceObjectRegistry.h"

namespace visionaray {

VSNRAY_FUNC
inline float epsilonFrom(const vec3 &P, const vec3 &dir, float t)
{
  constexpr float ulpEpsilon = 0x1.fp-18;
  return max_element(vec4(abs(P), max_element(abs(dir)) * t)) * ulpEpsilon;
}

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
  PrimitiveId,
  WorldPosition,
  ObjectPosition,
  Ng,
  Ns,
  Tangent,
  Bitangent,
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
#ifdef WITH_CUDA
  cuda_texture_ref<vector<4, unorm<8>>, 2> bgImage;
#elif defined(WITH_HIP)
  hip_texture_ref<vector<4, unorm<8>>, 2> bgImage;
#else
  texture_ref<vector<4, unorm<8>>, 2> bgImage;
#endif
  RenderMode renderMode{RenderMode::Default};
  int maxBounce{7};
  float4 *clipPlanes{nullptr};
  unsigned numClipPlanes{0};
  int sampleLimit{1024};
  int pixelSamples{1};
  int accumID{0};
  // TAA
  bool taaEnabled{false};
  float taaAlpha{0.3f};
  mat4 prevMV{mat4::identity()};
  mat4 prevPR{mat4::identity()};
  mat4 currMV{mat4::identity()};
  mat4 currPR{mat4::identity()};
  // Volume
  bool gradientShading{false};
  float volumeSamplingRateInv{2.0f};
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
inline uint32_t getSphereIndex(const dco::Array &indexArray, unsigned primID)
{
  uint32_t index;
  if (indexArray.len > 0) {
    index = ((uint32_t *)indexArray.data)[primID];
  } else {
    index = primID;
  }
  return index;
}

VSNRAY_FUNC
inline uint2 getConeIndex(const dco::Array &indexArray, unsigned primID)
{
  uint2 index;
  if (indexArray.len > 0) {
    index = ((uint2 *)indexArray.data)[primID];
  } else {
    index = uint2(primID * 2, primID * 2 + 1);
  }
  return index;
}

VSNRAY_FUNC
inline uint2 getCylinderIndex(const dco::Array &indexArray, unsigned primID)
{
  uint2 index;
  if (indexArray.len > 0) {
    index = ((uint2 *)indexArray.data)[primID];
  } else {
    index = uint2(primID * 2, primID * 2 + 1);
  }
  return index;
}

VSNRAY_FUNC
inline uint3 getTriangleIndex(const dco::Array &indexArray, unsigned primID)
{
  uint3 index;
  if (indexArray.len > 0) {
    index = ((uint3 *)indexArray.data)[primID];
  } else {
    index = uint3(primID * 3, primID * 3 + 1, primID * 3 + 2);
  }
  return index;
}

VSNRAY_FUNC
inline uint4 getQuadIndex(const dco::Array &indexArray, unsigned primID)
{
  uint4 index;
  if (indexArray.len > 0) {
    index = ((uint4 *)indexArray.data)[primID/2]; // primID refers to triangles!
  } else {
    primID /= 2; // tri to quad
    index = uint4(primID * 4, primID * 4 + 1, primID * 4 + 2, primID * 4 + 3);
  }
  return index;
}

VSNRAY_FUNC
inline void getNormals(const dco::Geometry &geom,
                       unsigned primID,
                       const vec3 hitPos,
                       const vec2 uv,
                       vec3 &Ng,
                       vec3 &Ns)
{
  // TODO: doesn't work for instances yet
  if (geom.type == dco::Geometry::Triangle) {
    auto tri = geom.as<dco::Triangle>(primID);
    Ng = normalize(cross(tri.e1,tri.e2));
    if (geom.faceVarying.normal.len
        && geom.faceVarying.normal.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
      uint3 index(3 * primID, 3 * primID + 1, 3 * primID + 2);
      auto *normals = (const vec3 *)geom.vertex.normal.data;
      vec3 n1 = normals[index.x];
      vec3 n2 = normals[index.y];
      vec3 n3 = normals[index.z];
      Ns = lerp_r(n1, n2, n3, uv.x, uv.y);
      Ns = normalize(Ns);
    } else if (geom.vertex.normal.len
        && geom.vertex.normal.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
      uint3 index = getTriangleIndex(geom.index, primID);
      auto *normals = (const vec3 *)geom.vertex.normal.data;
      vec3 n1 = normals[index.x];
      vec3 n2 = normals[index.y];
      vec3 n3 = normals[index.z];
      Ns = lerp_r(n1, n2, n3, uv.x, uv.y);
      Ns = normalize(Ns);
    } else {
      Ns = Ng;
    }
  } else if (geom.type == dco::Geometry::Quad) {
    auto qtri = geom.as<dco::Triangle>(primID);
    Ng = normalize(cross(qtri.e1,qtri.e2));
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Sphere) {
    auto sph = geom.as<dco::Sphere>(primID);
    Ng = normalize((hitPos-sph.center) / sph.radius);
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Cone) {
    // reconstruct normal (see https://iquilezles.org/articles/intersectors/)
    auto cone = geom.as<dco::Cone>(primID);
    const vec3f ba = cone.v2 - cone.v1;
    const float m0 = dot(ba,ba);
    if (uv.x <= 0.f) {
      Ng = -ba*rsqrt(m0);
    } else if (uv.x >= 1) {
      Ng = ba*rsqrt(m0);
    } else {
      const float ra = cone.r1;
      const float rr = cone.r1 - cone.r2;
      const float hy = m0 + rr*rr;
      const float y = uv.y; // uv.y stores the unnormalized cone parameter t!
      const vec3f localPos = hitPos-cone.v1;
      Ng = normalize(m0*(m0*localPos+rr*ba*ra)-ba*hy*y);
    }
    Ns = Ng;
  } else if (geom.type == dco::Geometry::Cylinder) {
    auto cyl = geom.as<dco::Cylinder>(primID);
    vec3f axis = normalize(cyl.v2-cyl.v1);
    if (length(hitPos-cyl.v1) < cyl.radius)
      Ng = -axis;
    else if (length(hitPos-cyl.v2) < cyl.radius)
      Ng = axis;
    else {
      float t = dot(hitPos-cyl.v1, axis);
      vec3f pt = cyl.v1 + t * axis;
      Ng = normalize(hitPos-pt);
    }
    Ns = Ng;
  } else if (geom.type == dco::Geometry::BezierCurve) {
    float t = uv.x;
    vec3f curvePos = geom.as<dco::BezierCurve>(primID).f(t);
    Ng = normalize(hitPos-curvePos);
    Ns = Ng;
  } else if (geom.type == dco::Geometry::ISOSurface) {
    const auto &sf = geom.as<dco::ISOSurface>(0).field;
    float3 delta(sf.cellSize, sf.cellSize, sf.cellSize);
    delta *= float3(sf.voxelSpaceTransform(0,0),
                    sf.voxelSpaceTransform(1,1),
                    sf.voxelSpaceTransform(2,2));
    if (!sampleGradient(sf,sf.pointToVoxelSpace(hitPos),delta,Ng)) {
      Ng = vec3f(0.f);
    } else {
      Ng = normalize(Ng);
    }
    Ns = Ng;
  }
}

VSNRAY_FUNC
inline vec4 getTangent(
    const dco::Geometry &geom, unsigned primID, const vec3 hitPos, const vec2 uv)
{
  vec4f tng(0.f);

  if (geom.type == dco::Geometry::Triangle) {
    if (geom.faceVarying.tangent.len) {
      uint3 index(3 * primID, 3 * primID + 1, 3 * primID + 2);
      if (geom.faceVarying.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
        auto *tangents = (const vec3 *)geom.faceVarying.tangent.data;
        vec3 tng1 = tangents[index.x];
        vec3 tng2 = tangents[index.y];
        vec3 tng3 = tangents[index.z];
        tng = vec4(lerp_r(tng1, tng2, tng3, uv.x, uv.y), 1.f);
      } else if (geom.faceVarying.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC4) {
        auto *tangents = (const vec4 *)geom.faceVarying.tangent.data;
        vec4 tng1 = tangents[index.x];
        vec4 tng2 = tangents[index.y];
        vec4 tng3 = tangents[index.z];
        tng = lerp_r(tng1, tng2, tng3, uv.x, uv.y);
      }
    } else if (geom.vertex.tangent.len) {
      uint3 index = getTriangleIndex(geom.index, primID);
      if (geom.vertex.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC3) {
        auto *tangents = (const vec3 *)geom.vertex.tangent.data;
        vec3 tng1 = tangents[index.x];
        vec3 tng2 = tangents[index.y];
        vec3 tng3 = tangents[index.z];
        tng = vec4(lerp_r(tng1, tng2, tng3, uv.x, uv.y), 1.f);
      } else if (geom.vertex.tangent.typeInfo.dataType == ANARI_FLOAT32_VEC4) {
        auto *tangents = (const vec4 *)geom.vertex.tangent.data;
        vec4 tng1 = tangents[index.x];
        vec4 tng2 = tangents[index.y];
        vec4 tng3 = tangents[index.z];
        tng = lerp_r(tng1, tng2, tng3, uv.x, uv.y);
      }
    }
  }

  return tng;
}

VSNRAY_FUNC
inline vec4 getAttribute(const dco::Geometry &geom,
                         const dco::Instance &inst,
                         dco::Attribute attrib,
                         unsigned primID,
                         const vec2 uv)
{
  vec4f color{0.f, 0.f, 0.f, 1.f};

  if (attrib == dco::Attribute::None)
    return color;

  if ((int)attrib >= 5) // hit attributes!
    return color;

  dco::Array faceVaryingColors = geom.faceVarying.attributes[(int)attrib];
  dco::Array vertexColors = geom.vertex.attributes[(int)attrib];
  dco::Array primitiveColors = geom.primitiveAttributes[(int)attrib];
  dco::Uniform geometryColor = geom.uniformAttributes[(int)attrib];
  dco::Uniform instanceColor = inst.uniformAttributes[(int)attrib];

  const TypeInfo &faceVaryingColorInfo = faceVaryingColors.typeInfo;
  const TypeInfo &vertexColorInfo = vertexColors.typeInfo;
  const TypeInfo &primitiveColorInfo = primitiveColors.typeInfo;

  // vertex colors take precedence over primitive colors
  if (faceVaryingColors.len > 0) {
    if (geom.type == dco::Geometry::Triangle) {
      uint3 index(3 * primID,
                  3 * primID + 1,
                  3 * primID + 2);
      const auto *source1
          = (const uint8_t *)faceVaryingColors.data
              + index.x * faceVaryingColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)faceVaryingColors.data
              + index.y * faceVaryingColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)faceVaryingColors.data
              + index.z * faceVaryingColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, faceVaryingColorInfo);
      vec4f c2 = toRGBA(source2, faceVaryingColorInfo);
      vec4f c3 = toRGBA(source3, faceVaryingColorInfo);
      color = lerp_r(c1, c2, c3, uv.x, uv.y);
    }
    else if (geom.type == dco::Geometry::Quad) {
      uint4 index(4 * primID,
                  4 * primID + 1,
                  4 * primID + 2,
                  4 * primID + 3);
      const auto *source1
          = (const uint8_t *)faceVaryingColors.data
              + index.x * faceVaryingColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)faceVaryingColors.data
              + index.y * faceVaryingColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)faceVaryingColors.data
              + index.z * faceVaryingColorInfo.sizeInBytes;
      const auto *source4
          = (const uint8_t *)faceVaryingColors.data
              + index.w * faceVaryingColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, faceVaryingColorInfo);
      vec4f c2 = toRGBA(source2, faceVaryingColorInfo);
      vec4f c3 = toRGBA(source3, faceVaryingColorInfo);
      vec4f c4 = toRGBA(source4, faceVaryingColorInfo);
      if (primID%2==0)
        color = lerp_r(c1, c2, c4, uv.x, uv.y);
      else
        color = lerp_r(c3, c4, c2, 1.f-uv.x, 1.f-uv.y);
    }
  } else if (vertexColors.len > 0) {
    if (geom.type == dco::Geometry::Triangle) {
      uint3 index = getTriangleIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)vertexColors.data
              + index.z * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      vec4f c3 = toRGBA(source3, vertexColorInfo);
      color = lerp_r(c1, c2, c3, uv.x, uv.y);
    }
    else if (geom.type == dco::Geometry::Quad) {
      uint4 index = getQuadIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      const auto *source3
          = (const uint8_t *)vertexColors.data
              + index.z * vertexColorInfo.sizeInBytes;
      const auto *source4
          = (const uint8_t *)vertexColors.data
              + index.w * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      vec4f c3 = toRGBA(source3, vertexColorInfo);
      vec4f c4 = toRGBA(source4, vertexColorInfo);
      if (primID%2==0)
        color = lerp_r(c1, c2, c4, uv.x, uv.y);
      else
        color = lerp_r(c3, c4, c2, 1.f-uv.x, 1.f-uv.y);
    }
    else if (geom.type == dco::Geometry::Sphere) {
      uint32_t index = getSphereIndex(geom.index, primID);
      const auto *source
          = (const uint8_t *)vertexColors.data
              + index * vertexColorInfo.sizeInBytes;
      color = toRGBA(source, vertexColorInfo);
    }
    else if (geom.type == dco::Geometry::Cone) {
      uint2 index = getConeIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      color = lerp_r(c1, c2, uv.x);
    }
    else if (geom.type == dco::Geometry::Cylinder) {
      uint2 index = getCylinderIndex(geom.index, primID);
      const auto *source1
          = (const uint8_t *)vertexColors.data
              + index.x * vertexColorInfo.sizeInBytes;
      const auto *source2
          = (const uint8_t *)vertexColors.data
              + index.y * vertexColorInfo.sizeInBytes;
      vec4f c1 = toRGBA(source1, vertexColorInfo);
      vec4f c2 = toRGBA(source2, vertexColorInfo);
      color = lerp_r(c1, c2, uv.x);
    }
  } else if (primitiveColors.len > 0) {
    const auto *source
        = (const uint8_t *)primitiveColors.data
            + primID * primitiveColorInfo.sizeInBytes;
    color = toRGBA(source, primitiveColorInfo);
  } else if (geometryColor.isSet) {
    color = geometryColor.value;
  } else if (instanceColor.isSet) {
    color = instanceColor.value;
  }

  return color;
}

VSNRAY_FUNC
inline dco::AttributeRec getAttributes(const dco::Geometry &geom,
                                       const dco::Instance &inst,
                                       float3 worldPos,
                                       float3 worldNormal,
                                       float3 objectPos,
                                       float3 objectNormal,
                                       unsigned primID,
                                       const vec2 uv)
{
  dco::AttributeRec res;
  res._0 = getAttribute(geom, inst, dco::Attribute::_0, primID, uv);
  res._1 = getAttribute(geom, inst, dco::Attribute::_1, primID, uv);
  res._2 = getAttribute(geom, inst, dco::Attribute::_2, primID, uv);
  res._3 = getAttribute(geom, inst, dco::Attribute::_3, primID, uv);
  res.color = getAttribute(geom, inst, dco::Attribute::Color, primID, uv);
  // hit attributes:
  res.worldPos = float4(worldPos,1.f);
  res.worldNormal = float4(worldNormal,1.f);
  res.objectPos = float4(objectPos,1.f);
  res.objectNormal = float4(objectNormal,1.f);
  return res;
}

VSNRAY_FUNC
inline vec4 getSample(const dco::Sampler &samp,
                      const DeviceObjectRegistry &onDevice,
                      const dco::AttributeRec &attribs,
                      float3 objPos,
                      unsigned primID)
{
  if (samp.type == dco::Sampler::Primitive) {
    const TypeInfo &info = samp.asPrimitive.typeInfo;
    const auto *source = samp.asPrimitive.data
        + (primID * info.sizeInBytes) + (samp.asPrimitive.offset * info.sizeInBytes);
    return toRGBA(source, info);
  } else if (samp.type == dco::Sampler::Transform) {
    vec4f inAttr = attribs.get(samp.inAttribute);
    return samp.outTransform * inAttr + samp.outOffset;
  } else if (samp.type == dco::Sampler::Volume) {
    vec4f inPos(objPos,1.f);
    inPos = samp.inTransform * inPos + samp.inOffset;
    //std::cout << inPos << '\n';
    const auto &vol = onDevice.volumes[samp.asVolume.volID];
    const auto &sf  = vol.field;
    const auto &P = sf.pointToVoxelSpace(inPos.xyz());
    float v = 0.f;
    vec4f s{0.f, 0.f, 0.f, 1.f};
    if (sampleField(sf,P,v)) {
      // TODO: support other volume types:
      s = postClassify(vol.asTransferFunction1D,v);
    }
    return samp.outTransform * s + samp.outOffset;
  } else {
    vec4f inAttr = attribs.get(samp.inAttribute);

    inAttr = samp.inTransform * inAttr + samp.inOffset;

    vec4f s{0.f, 0.f, 0.f, 1.f};

    if (samp.type == dco::Sampler::Image1D)
      s = tex1D(samp.asImage1D, inAttr.x);
    else if (samp.type == dco::Sampler::Image2D)
      s = tex2D(samp.asImage2D, inAttr.xy());
    else if (samp.type == dco::Sampler::Image3D)
      s = tex3D(samp.asImage3D, inAttr.xyz());

    return samp.outTransform * s + samp.outOffset;
  }
}

VSNRAY_FUNC
inline vec4 getRGBA(const dco::MaterialParamRGB &param,
                    const DeviceObjectRegistry &onDevice,
                    const dco::AttributeRec &attribs,
                    float3 objPos,
                    unsigned primID)
{
  if (param.samplerID < UINT_MAX)
    return getSample(
        onDevice.samplers[param.samplerID], onDevice, attribs, objPos, primID);
  else if (param.attribute != dco::Attribute::None)
    return attribs.get(param.attribute);
  else
    return vec4f(param.rgb, 1.f);
}

VSNRAY_FUNC
inline vec2 getUV(const dco::MaterialParamUV &param,
                  const DeviceObjectRegistry &onDevice,
                  const dco::AttributeRec &attribs,
                  float3 objPos,
                  unsigned primID)
{
  if (param.samplerID < UINT_MAX)
    return getSample(
        onDevice.samplers[param.samplerID], onDevice, attribs, objPos, primID).xy();
  else if (param.attribute != dco::Attribute::None)
    return attribs.get(param.attribute).xy();
  else
    return param.uv;
}

VSNRAY_FUNC
inline float getF(const dco::MaterialParamF &param,
                  const DeviceObjectRegistry &onDevice,
                  const dco::AttributeRec &attribs,
                  float3 objPos,
                  unsigned primID)
{
  if (param.samplerID < UINT_MAX)
    return getSample(
        onDevice.samplers[param.samplerID], onDevice, attribs, objPos, primID).x;
  else if (param.attribute != dco::Attribute::None)
    return attribs.get(param.attribute).x;
  else
    return param.f;
}

VSNRAY_FUNC
inline vec4 getColorMatte(const dco::Material &mat,
                          const DeviceObjectRegistry &onDevice,
                          const dco::AttributeRec &attribs,
                          float3 objPos,
                          unsigned primID)
{
  return getRGBA(mat.asMatte.color, onDevice, attribs, objPos, primID);
}

VSNRAY_FUNC
inline vec4 getColorPBM(const dco::Material &mat,
                        const DeviceObjectRegistry &onDevice,
                        const dco::AttributeRec &attribs,
                        float3 objPos,
                        unsigned primID)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, onDevice, attribs, objPos, primID);
  vec4f color = getRGBA(
      mat.asPhysicallyBased.baseColor, onDevice, attribs, objPos, primID);
  return lerp_r(color, vec4f(0.f, 0.f, 0.f, color.w), metallic);
}

VSNRAY_FUNC
inline vec4 getColor(const dco::Material &mat,
                     const DeviceObjectRegistry &onDevice,
                     const dco::AttributeRec &attribs,
                     float3 objPos,
                     unsigned primID)
{
  vec4f color{0.f, 0.f, 0.f, 1.f};
  if (mat.type == dco::Material::Matte)
    color = getColorMatte(mat, onDevice, attribs, objPos, primID);
  else if (mat.type == dco::Material::PhysicallyBased) {
    color = getColorPBM(mat, onDevice, attribs, objPos, primID);
  }
  return color;
}

VSNRAY_FUNC
inline float getOpacity(const dco::Material &mat,
                        const DeviceObjectRegistry &onDevice,
                        const dco::AttributeRec &attribs,
                        float3 objPos,
                        unsigned primID)
{
  float opacity = 1.f;
  dco::AlphaMode mode{dco::AlphaMode::Opaque};
  float cutoff = 0.5f;

  if (mat.type == dco::Material::Matte) {
    vec4f color = getColorMatte(mat, onDevice, attribs, objPos, primID);
    opacity = color.w * getF(mat.asMatte.opacity, onDevice, attribs, objPos, primID);
    mode = mat.asMatte.alphaMode;
    cutoff = mat.asMatte.alphaCutoff;
  } else if (mat.type == dco::Material::PhysicallyBased) {
    vec4f color = getColorPBM(mat, onDevice, attribs, objPos, primID);
    opacity = color.w *
        getF(mat.asPhysicallyBased.opacity, onDevice, attribs, objPos, primID);
    mode = mat.asPhysicallyBased.alphaMode;
    cutoff = mat.asPhysicallyBased.alphaCutoff;
  }

  if (mode == dco::AlphaMode::Opaque)
    return 1.f;
  else if (mode == dco::AlphaMode::Blend)
    return opacity;
  else // mode==Mask
    return opacity >= cutoff ? 1.f : 0.f;
}

VSNRAY_FUNC
inline float getTransmission(const dco::Material &mat,
                             const DeviceObjectRegistry &onDevice,
                             const dco::AttributeRec &attribs,
                             float3 objPos,
                             unsigned primID)
{
  if (mat.type == dco::Material::PhysicallyBased) {
    return getF(mat.asPhysicallyBased.transmission, onDevice, attribs, objPos, primID);
  } else {
    return 0.f;
  }
}

VSNRAY_FUNC
inline float getIOR(const dco::Material &mat)
{
  if (mat.type == dco::Material::PhysicallyBased) {
    return mat.asPhysicallyBased.ior;
  } else {
    return 1.f;
  }
}

VSNRAY_FUNC
inline vec3 getPerturbedNormal(const dco::Material &mat,
                               const DeviceObjectRegistry &onDevice,
                               const dco::AttributeRec &attribs,
                               float3 objPos,
                               unsigned primID,
                               const vec3 T, const vec3 B, const vec3 N)
{
  vec3f pn = N;

  mat3 TBN(T,B,N);
  if (mat.type == dco::Material::PhysicallyBased) {
    if (onDevice.samplers && dco::validHandle(mat.asPhysicallyBased.normal.samplerID)) {
      const auto &samp = onDevice.samplers[mat.asPhysicallyBased.normal.samplerID];
      vec4 s = getSample(samp, onDevice, attribs, objPos, primID);
      vec3 tbnN = s.xyz();
      if (length(tbnN) > 0.f) {
        vec3f objN = normalize(TBN * tbnN);
        //pn = lerp_r(N, objN, 0.5f); // encode in outTransform!
        pn = objN;
      }
    }
  }

  return pn;
}

VSNRAY_FUNC
inline mat3 getNormalTransform(const dco::Instance &inst, const Ray &ray)
{
  if (inst.type == dco::Instance::Transform) {
    return inst.normalXfms[0];
  } else if (inst.type == dco::Instance::MotionTransform) {

    float rayTime = clamp(ray.time, inst.time.min, inst.time.max);

    float time01 = rayTime - inst.time.min / (inst.time.max - inst.time.min);

    unsigned ID1 = unsigned(float(inst.len-1) * time01);
    unsigned ID2 = min((unsigned)inst.len-1, ID1+1);

    float frac = time01 * (inst.len-1) - ID1;

    return lerp_r(inst.normalXfms[ID1],
                  inst.normalXfms[ID2],
                  frac);
  }

  return {};
}

//=========================================================
// BSDF eval
//=========================================================

VSNRAY_FUNC
inline float pow2(float f)
{
  return f*f;
}

VSNRAY_FUNC
inline float pow5(float f)
{
  return f*f*f*f*f;
}

// From: https://google.github.io/filament/Filament.html
VSNRAY_FUNC
inline vec3 F_Schlick(float u, vec3 f0)
{
  return f0 + (vec3f(1.f) - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float F_Schlick(float u, float f0)
{
  return f0 + (1.f - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float F_Schlick(float u, float f0, float f90)
{
  return f0 + (f90 - f0) * pow5(1.f - u);
}

VSNRAY_FUNC
inline float Fd_Lambert()
{
  return constants::inv_pi<float>();
}

VSNRAY_FUNC
inline float Fd_Burley(float NdotV, float NdotL, float LdotH, float roughness)
{
  float f90 = 0.5f + 2.f * roughness * LdotH * LdotH;
  float lightScatter = F_Schlick(NdotL, 1.f, f90);
  float viewScatter = F_Schlick(NdotV, 1.f, f90);
  return lightScatter * viewScatter * constants::inv_pi<float>();
}

VSNRAY_FUNC
inline float D_GGX(float NdotH, float roughness, float EPS)
{
  float alpha = roughness;
  float alpha2 = alpha * alpha;

  float d = (NdotH * alpha2 - NdotH) * NdotH + 1.f;
  float denom = constants::pi<float>() * d * d;
  return alpha2 / fmaxf(EPS, denom);
  //float denom
  //  = constants::pi<float>()*pow2(NdotH*NdotH*(alpha*alpha-1.f)+1.f);
  //if (denom==0.f) denom = EPS;
  //return (alpha*alpha*heaviside(NdotH)) / denom;
}

VSNRAY_FUNC
inline float D_GGX_Anisotropic(float NdotH, float TdotH, float BdotH, float at, float ab)
{
  float a2 = at*ab;
  float3 v(ab * TdotH, at * BdotH, a2 * NdotH);
  float v2 = fmaxf(dot(v,v),1e-14f);
  float w2 = a2 / v2;
  return a2 * w2 * w2 * constants::inv_pi<float>();
}

VSNRAY_FUNC
inline float G_SmithGGX(float NdotL, float NdotV, float roughness)
{
  float alpha = roughness;
  return ((2.f * NdotL) / (NdotL + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotL*NdotL)))
    *    ((2.f * NdotV) / (NdotV + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotV*NdotV)));
}

VSNRAY_FUNC
inline float V_SmithGGX(float NdotV, float NdotL, float roughness)
{
#if 0
  // full variant, equivalent to G/(4*(n*v)*(n*l))
  constexpr float EPS = 1e-14f;
  float alpha = roughness;
  float denom = 4.f * NdotV * NdotL;
  return G_SmithGGX(NdotL, NdotV, roughness) / max(EPS,denom);
#else
  // equivalent but simplified:
  float alpha = roughness;
  float GGXV = 1.f / (NdotV + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotV*NdotV));
  float GGXL = 1.f / (NdotL + sqrtf(alpha*alpha + (1.f-alpha*alpha) * NdotL*NdotL));
  return 0.5f / (GGXV + GGXL);
#endif
}

VSNRAY_FUNC
inline float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
  // height-correlated Smith function - accorcing to Heitz
  // correlating masking and shading is a bit more accurate:
  float alpha = roughness;
  float a2 = alpha * alpha;
  float GGXV = NdotL * sqrtf(a2 + (1.f - a2) * NdotV * NdotV);
  float GGXL = NdotV * sqrtf(a2 + (1.f - a2) * NdotL * NdotL);
  return 0.5f / (GGXV + GGXL);
}

VSNRAY_FUNC
inline float V_Kelemen(float LdotH, const float EPS)
{
  return 0.25f / fmaxf(EPS, (LdotH * LdotH));
}

VSNRAY_FUNC
inline float mapRoughness(float roughness)
{
  return fmaxf(0.045f, roughness);
}

VSNRAY_FUNC
inline void pdfGGXVNDF(const float3 &lightDir,
                       const float3 &viewDir,
                       bool anisotropicSpecular,
                       float alpha_x, float alpha_y,
                       float eta_i, float eta_t,
                       const vec3 Ng, const vec3 Ns,
                       // tangent/bitangent transformed by anisotropic direction:
                       const vec3 anisotropicT, const vec3 anisotropicB,
                       float &pdfSpec,
                       float &pdfTrans)
{
  constexpr float EPS = 1e-14f;

  // calculate specular PDF:
  pdfSpec = 0.f, pdfTrans = 0.f;
  const float NdotL = dot(Ns,lightDir);
  const float NdotV = fmaxf(EPS,dot(Ns,viewDir));

  if (NdotL > 0.f) {
    float3 H = viewDir+lightDir;
    float lenH = length(H);

    if (lenH > 1e-4f) {
      H /= lenH;

      float NdotH = fmaxf(EPS,dot(Ns,H));
      float VdotH = fmaxf(EPS,dot(viewDir,H));

      float D, G1;

      if (!anisotropicSpecular) {
        const float alpha_x2 = alpha_x * alpha_x;
        D = D_GGX(NdotH, alpha_x, EPS);
        G1 = 2.f*NdotV / (NdotV+sqrtf(alpha_x2 + (1.f-alpha_x2) * NdotV * NdotV));
      } else {
        const float alpha_x2 = alpha_x * alpha_x;
        const float alpha_y2 = alpha_y * alpha_y;
        const float TdotH = dot(anisotropicT,H);
        const float BdotH = dot(anisotropicB,H);
        D = D_GGX_Anisotropic(NdotH, TdotH, BdotH, alpha_x, alpha_y);

        const float TdotV = dot(anisotropicT,viewDir);
        const float BdotV = dot(anisotropicB,viewDir);

        const float alpha_v = sqrtf(TdotV * TdotV * alpha_x2 + BdotV * BdotV * alpha_y2);
        const float alpha_v2 = alpha_v * alpha_v;
        G1 = 2.f*NdotV / (NdotV+sqrtf(alpha_v2 + (1.f-alpha_v2) * NdotV * NdotV));
      }

      pdfSpec = (D*G1) / (4.f*NdotV);
    }
  }

  // calculate transmissive PDF:
  else if (NdotL < 0.f) {
    float3 H = -(viewDir * eta_i + lightDir * eta_t);
    float lenH = length(H);

    if (lenH > 1e-4f) {
      H /= lenH;

      if (dot(Ns,H) < 0.f) H = -H;

      const float alpha = alpha_x; // TODO!
      const float alpha2 = alpha * alpha;

      float NdotH = fmaxf(EPS,dot(Ns,H));
      float LdotH = fabsf(dot(lightDir,H));
      float VdotH = fabsf(dot(viewDir,H));

      if (VdotH > 0.f && LdotH > 0.f) {
        float D = D_GGX(NdotH, alpha, EPS);
        float G1 = 2.f*NdotV / (NdotV+sqrtf(alpha2 + (1.f-alpha2) * NdotV * NdotV));
        float pdfNe = (D*G1 * VdotH) / NdotV;

        float denom = (eta_i * VdotH + eta_t * LdotH);
        float jacobian = (eta_t * eta_t * LdotH) / (denom * denom);

        pdfTrans = pdfNe * jacobian;
      }
    }
  }
}

VSNRAY_FUNC
inline void pdfGGXVNDF(const float3 &lightDir,
                       const float3 &viewDir,
                       float alpha, float eta_i, float eta_t,
                       const vec3 Ng, const vec3 Ns,
                       float &pdfSpec)
{
  // convenience overload, isotropic and w/o transmissive pdf:
  float pdfTrans{0.f};
  float3 T{0.f}, B{0.f};
  pdfGGXVNDF(lightDir,
             viewDir,
             false,
             alpha, alpha,
             eta_i, eta_t,
             Ng, Ns,
             T, B,
             pdfSpec,
             pdfTrans);
}

VSNRAY_FUNC
inline vec3 evalMatteMaterial(const dco::Material &mat,
                              const DeviceObjectRegistry &onDevice,
                              const dco::AttributeRec &attribs,
                              float3 objPos,
                              unsigned primID,
                              const vec3 Ng, const vec3 Ns,
                              const vec3 T, const vec3 B,
                              const vec3 viewDir, const vec3 lightDir,
                              float *pdf = nullptr)
{
  const vec3 color = getColorMatte(mat, onDevice, attribs, objPos, primID).xyz();
  vec3 diffuseBRDF = color * Fd_Lambert();
  if (pdf != nullptr) {
    *pdf = fmaxf(0.f,dot(Ns,lightDir)) * constants::inv_pi<float>();
  }
  return diffuseBRDF;
}

VSNRAY_FUNC
inline vec3 evalPhysicallyBasedMaterial(const dco::Material &mat,
                                        const DeviceObjectRegistry &onDevice,
                                        const dco::AttributeRec &attribs,
                                        float3 objPos,
                                        unsigned primID,
                                        const vec3 Ng, const vec3 Ns,
                                        const vec3 T, const vec3 B,
                                        const vec3 viewDir, const vec3 lightDir,
                                        float *pdf = nullptr)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, onDevice, attribs, objPos, primID);
  const float roughness = getF(
      mat.asPhysicallyBased.roughness, onDevice, attribs, objPos, primID);

  const float anisotropyStrength = getF(
      mat.asPhysicallyBased.anisotropyStrength, onDevice, attribs, objPos, primID);
  const float2 anisotropyDirection = getUV(
      mat.asPhysicallyBased.anisotropyDirection, onDevice, attribs, objPos, primID);
  const float anisotropyRotation = getF(
      mat.asPhysicallyBased.anisotropyRotation, onDevice, attribs, objPos, primID);

  const float clearcoat = getF(
      mat.asPhysicallyBased.clearcoat, onDevice, attribs, objPos, primID);
  const float clearcoatRoughness = getF(
      mat.asPhysicallyBased.clearcoatRoughness, onDevice, attribs, objPos, primID);

  const float transmission = getF(
      mat.asPhysicallyBased.transmission, onDevice, attribs, objPos, primID);

  const float ior = mat.asPhysicallyBased.ior;

  vec3 sheenColor = getRGBA(
      mat.asPhysicallyBased.sheenColor, onDevice, attribs, objPos, primID).xyz();
  const float sheenRoughness = getF(
      mat.asPhysicallyBased.sheenRoughness, onDevice, attribs, objPos, primID);

  const float perceptualRoughness = mapRoughness(roughness);
  const float perceptualClearcoatRoughness = mapRoughness(clearcoatRoughness);
  const float perceptualSheenRoughness = mapRoughness(sheenRoughness);

  const float alpha = perceptualRoughness * perceptualRoughness;
  const float clearcoatAlpha = perceptualClearcoatRoughness * perceptualClearcoatRoughness;

  // anisotropic basis:
  const float2 rotation(cosf(anisotropyRotation), sinf(anisotropyRotation));
  float2 direction = anisotropyDirection;
  direction = mat2(rotation.x, rotation.y, -rotation.y, rotation.x) * normalize(direction);
  const mat3 TBN(T,B,Ns);
  const float3 anisotropicT = normalize(TBN * float3(direction, 0.f));
  const float3 anisotropicB = normalize(cross(Ns, anisotropicT));

  constexpr float EPS = 1e-14f;
  const vec3 H = normalize(lightDir+viewDir);
  const float NdotV = fabsf(dot(Ns,viewDir)) + EPS;
  const float NdotH = fmaxf(EPS,dot(Ns,H));
  const float NdotL = fmaxf(EPS,dot(Ns,lightDir));
  const float VdotH = fmaxf(EPS,dot(viewDir,H));
  const float LdotH = fmaxf(EPS,dot(lightDir,H));
  const float TdotH = dot(anisotropicT,H);
  const float BdotH = dot(anisotropicB,H);

  // Get baseColor:
  vec3 baseColor = getRGBA(
      mat.asPhysicallyBased.baseColor, onDevice, attribs, objPos, primID).xyz();

  // Metallic materials don't reflect diffusely:
  vec3 diffuseColor = lerp_r(baseColor, vec3f(0.f), metallic);

  // Fresnel
  vec3 f0 = lerp_r(vec3(pow2((1.f-ior)/(1.f+ior))), baseColor, metallic);
  vec3 F = F_Schlick(VdotH, f0);

  // Diffuse:
//vec3 diffuseBRDF = (1.f-F) * diffuseColor * Fd_Lambert();
  vec3 diffuseBRDF = (1.f-F) * diffuseColor * Fd_Burley(NdotV, NdotL, LdotH, alpha);

  bool entering = dot(Ns,viewDir) > 0.f;
  float eta_i = entering ? 1.0f : ior;
  float eta_t = entering ? ior : 1.0f;

  // GGX microfacet distribution
  bool anisotropicSpecular = false;
  float at = alpha, ab = alpha;
  float D = 0.f;
  if (anisotropyStrength > 0.f && length(anisotropicT) > EPS
        && length(anisotropicB) > EPS) {
    anisotropicSpecular = true;
    float aspect = sqrtf(1.f - anisotropyStrength * 0.9f);
    at = max(alpha / aspect, 0.001f);
    ab = max(alpha * aspect, 0.001f);
    D = D_GGX_Anisotropic(NdotH, TdotH, BdotH, at, ab);
  } else {
    D = D_GGX(NdotH, alpha, EPS);
  }

  // Masking-shadowing term integrated (and simplified into) V
  // also allows us to toy with different variants of V
//float V = V_SmithGGX(NdotV, NdotL, alpha);
  float V = V_SmithGGXCorrelated(NdotV, NdotL, alpha);
  vec3 specularBRDF = F * D * V;

  // Clearcoat
  float Dc = D_GGX(NdotH, clearcoatAlpha, EPS);
  float Vc = V_Kelemen(LdotH, EPS);
  float Fc = F_Schlick(LdotH, 0.04f) * clearcoat;
  float Frc = (Dc * Vc) * Fc;

  // (Charlie) sheen
  float sheenAlphaInv = 1.f/perceptualSheenRoughness;
  float cos2h = NdotH * NdotH;
  float sin2h = fmaxf(1.f - cos2h, 0.0078125f);
  vec3 Ds = sheenColor *
      ((2.f + sheenAlphaInv) * powf(sin2h, sheenAlphaInv * 0.5f) / (constants::two_pi<float>()));

  if (pdf != nullptr) {
    const float fClear = clearcoat * 0.04f;
    const float baseEnergy = 1.f-fClear;

    const float fSpec = rgb_to_luminance(F);
    const float remainingEnergy = baseEnergy * (1.f-fSpec);

    float wDiff  = rgb_to_luminance(diffuseColor) * remainingEnergy * (1.f-transmission);
    float wSpec  = baseEnergy * fSpec;
    float wTrans = remainingEnergy * transmission;
    float wClear = fClear;
    float wSum   = wDiff + wSpec + wTrans + wClear;

    // TODO: CDF sampling if we have multiple lobes
    float pDiff  = wSum > 0.f ? wDiff  / wSum : 0.0f;
    float pSpec  = wSum > 0.f ? wSpec  / wSum : 0.333f;
    float pTrans = wSum > 0.f ? wTrans / wSum : 0.667f;
    float pClear = 1.f - pDiff - pSpec - pTrans;

    float pdfDiff = fmaxf(0.f,dot(Ns,lightDir)) * constants::inv_pi<float>();

    float pdfSpec = 0.f, pdfTrans = 0.f;
    pdfGGXVNDF(lightDir,
               viewDir,
               anisotropicSpecular,
               at, ab,
               eta_i, eta_t,
               Ng, Ns,
               anisotropicT, anisotropicB,
               pdfSpec,
               pdfTrans);

    float pdfClear = 0.f;
    float ceta_i = entering ? 1.0f : 1.5f;
    float ceta_t = entering ? 1.5f : 1.0f;
    pdfGGXVNDF(lightDir,
               viewDir,
               clearcoatAlpha,
               ceta_i,
               ceta_t,
               Ng, Ns,
               pdfClear);

    *pdf = pDiff*pdfDiff + pSpec*pdfSpec + pTrans*pdfTrans + pClear*pdfClear;
  }

  return ((diffuseBRDF + specularBRDF) * (1.f - Fc) + Frc) + Ds;
}

VSNRAY_FUNC
inline vec3 evalMaterial(const dco::Material &mat,
                         const DeviceObjectRegistry &onDevice,
                         const dco::AttributeRec &attribs,
                         float3 objPos,
                         unsigned primID,
                         const vec3 Ng, const vec3 Ns,
                         const vec3 T, const vec3 B,
                         const vec3 viewDir, const vec3 lightDir,
                         float *pdf = nullptr)
{
  vec3 materialColor{0.f, 0.f, 0.f};
  if (mat.type == dco::Material::Matte) {
    materialColor = evalMatteMaterial(mat,
                                      onDevice,
                                      attribs,
                                      objPos,
                                      primID,
                                      Ng, Ns,
                                      T, B,
                                      viewDir, lightDir,
                                      pdf);
  } else if (mat.type == dco::Material::PhysicallyBased) {
    materialColor = evalPhysicallyBasedMaterial(mat,
                                                onDevice,
                                                attribs,
                                                objPos,
                                                primID,
                                                Ng, Ns,
                                                T, B,
                                                viewDir, lightDir,
                                                pdf);
  }
  return materialColor;
}

//=========================================================
// BSDF sampling
//=========================================================

struct BSDFSample
{
  float3 dir;
  float3 f;
  float pdf;
  float cosT;
  bool isSpecular;
};

// Heitz 2018: Sampling the GGX Distribution of Visible Normals
VSNRAY_FUNC
inline float3 sampleGGXVNDF(float3 viewDir, // in local coordinates (0,0,1)
                            float alpha_x,
                            float alpha_y,
                            float U1,
                            float U2)
{
  // transforming the view direction to the hemisphere configuration
  float3 Vh = normalize(viewDir*float3(alpha_x,alpha_y,1.f));
  // orthonormal basis (with special case if cross product is zero)
  float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
  float3 T1 = lensq > 0.f ? float3(-Vh.y,Vh.x,0.f)/sqrtf(lensq) : float3(1,0,0);
  float3 T2 = cross(Vh,T1);
  // parameterization of the projected area
  float r = sqrtf(U1);
  float phi = constants::two_pi<float>() * U2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.f + Vh.z);
  t2 = (1.f-s) * sqrtf(fmaxf(0.f,1.f-t1*t1)) + s * t2;
  // reprojection onto hemisphere
  float3 Nh = t1*T1 + t2*T2 + sqrtf(1.f-t1*t1-t2*t2) * Vh;
  // transforming the normal back to the ellipsoid configuration
  float3 Ne = normalize(float3(Nh.x*alpha_x,Nh.y*alpha_y,fmaxf(0.f,Nh.z)));
  return Ne;
}

VSNRAY_FUNC
inline BSDFSample samplePhysicallyBasedMaterial(const dco::Material &mat,
                                                const DeviceObjectRegistry &onDevice,
                                                const dco::AttributeRec &attribs,
                                                vec3f objPos,
                                                unsigned primID,
                                                const vec3 Ng, const vec3 Ns,
                                                const vec3 T, const vec3 B,
                                                const vec3 viewDir,
                                                Random &rnd)
{
  const float metallic = getF(
      mat.asPhysicallyBased.metallic, onDevice, attribs, objPos, primID);
  const float roughness = getF(
      mat.asPhysicallyBased.roughness, onDevice, attribs, objPos, primID);

  const float anisotropyStrength = getF(
      mat.asPhysicallyBased.anisotropyStrength, onDevice, attribs, objPos, primID);
  const float2 anisotropyDirection = getUV(
      mat.asPhysicallyBased.anisotropyDirection, onDevice, attribs, objPos, primID);
  const float anisotropyRotation = getF(
      mat.asPhysicallyBased.anisotropyRotation, onDevice, attribs, objPos, primID);

  const float clearcoat = getF(
      mat.asPhysicallyBased.clearcoat, onDevice, attribs, objPos, primID);
  const float clearcoatRoughness = getF(
      mat.asPhysicallyBased.clearcoatRoughness, onDevice, attribs, objPos, primID);

  const float transmission = getF(
      mat.asPhysicallyBased.transmission, onDevice, attribs, objPos, primID);

  const float ior = mat.asPhysicallyBased.ior;

  vec3 sheenColor = getRGBA(
      mat.asPhysicallyBased.sheenColor, onDevice, attribs, objPos, primID).xyz();
  const float sheenRoughness = getF(
      mat.asPhysicallyBased.sheenRoughness, onDevice, attribs, objPos, primID);

  const float perceptualRoughness = mapRoughness(roughness);
  const float perceptualClearcoatRoughness = mapRoughness(clearcoatRoughness);

  const float alpha = perceptualRoughness * perceptualRoughness;
  const float clearcoatAlpha = perceptualClearcoatRoughness * perceptualClearcoatRoughness;

  // anisotropic basis:
  const float2 rotation(cosf(anisotropyRotation), sinf(anisotropyRotation));
  float2 direction = anisotropyDirection;
  direction = mat2(rotation.x, rotation.y, -rotation.y, rotation.x) * normalize(direction);
  const mat3 TBN(T,B,Ns);
  const float3 anisotropicT = normalize(TBN * float3(direction, 0.f));
  const float3 anisotropicB = normalize(cross(Ns, anisotropicT));

  constexpr float EPS = 1e-14f;
  const float NdotV = fmaxf(EPS,dot(Ns,viewDir));

  // Get baseColor:
  vec3 baseColor = getRGBA(
      mat.asPhysicallyBased.baseColor, onDevice, attribs, objPos, primID).xyz();

  // Metallic materials don't reflect diffusely:
  vec3 diffuseColor = lerp_r(baseColor, vec3f(0.f), metallic);

  // Fresnel
  vec3 f0 = lerp_r(vec3(pow2((1.f-ior)/(1.f+ior))), baseColor, metallic);
  vec3 F = F_Schlick(NdotV, f0);

  BSDFSample result;

  const float fClear = clearcoat * 0.04f;
  const float baseEnergy = 1.f-fClear;

  const float fSpec = rgb_to_luminance(F);
  const float remainingEnergy = baseEnergy * (1.f-fSpec);

  float wDiff  = rgb_to_luminance(diffuseColor) * remainingEnergy * (1.f-transmission);
  float wSpec  = baseEnergy * fSpec;
  float wTrans = remainingEnergy * transmission;
  float wClear = fClear;
  float wSum   = wDiff + wSpec + wTrans + wClear;

  // TODO: CDF sampling if we have multiple lobes
  float pDiff  = wSum > 0.f ? wDiff  / wSum : 0.0f;
  float pSpec  = wSum > 0.f ? wSpec  / wSum : 0.333f;
  float pTrans = wSum > 0.f ? wTrans / wSum : 0.667f;
  float pClear = 1.f - pDiff - pSpec - pTrans;

  auto w = faceforward(Ns, viewDir, Ng);
  auto v = fabsf(w.x) > fabsf(w.y) ? normalize(vec3(-w.z,0.f,w.x))
                                   : normalize(vec3(0.f,w.z,-w.y));
  auto u = cross(v,w);

  float lobe = rnd();

  // transform viewDir into local coordinate system
  mat3 basis(u,v,w);
  float3 V = transpose(basis)*viewDir;

  bool entering = dot(Ns,viewDir) > 0.f;
  float eta_i = entering ? 1.0f : ior;
  float eta_t = entering ? ior : 1.0f;

  float eta = entering ? 1.f/ior : ior;

  bool anisotropicSpecular = false;
  float at = alpha, ab = alpha;
  if (anisotropyStrength > 0.f && length(anisotropicT) > EPS
        && length(anisotropicB) > EPS) {
    anisotropicSpecular = true;
    float aspect = sqrtf(1.f - anisotropyStrength * 0.9f);
    at = max(alpha / aspect, 0.001f);
    ab = max(alpha * aspect, 0.001f);
  }

  result.isSpecular = false;
  if (lobe < pDiff) {
    auto sp = cosine_sample_hemisphere(rnd(),rnd());
    result.dir = normalize(sp.x*u+sp.y*v+sp.z*w);
  } else if (lobe < pDiff + pSpec) {
    // GGX sampling (Heitz 2018):
    float3 Ne = sampleGGXVNDF(V,at,ab,rnd(),rnd());
    // reflect along the microfacet normal transformed back to global space:
    result.dir = reflect(viewDir,basis*Ne);
  } else if (lobe < pDiff + pSpec + pTrans) {
    float3 Ne = sampleGGXVNDF(V,alpha,alpha,rnd(),rnd());
    if (dot(V,Ne) < 0.f) Ne = -Ne;
    result.dir = refract(viewDir,basis*Ne,eta_t/eta_i);
    if (length(result.dir) < 1e-4f) { // TIR
      result.dir = reflect(viewDir,basis*Ne);
    }
  } else {
    float3 Ne = sampleGGXVNDF(V,clearcoatAlpha,clearcoatAlpha,rnd(),rnd());
    result.dir = reflect(viewDir,basis*Ne);
  }

  float pdfDiff = fmaxf(0.f,dot(Ns,result.dir)) * constants::inv_pi<float>();

  float pdfSpec = 0.f, pdfTrans = 0.f;
  pdfGGXVNDF(result.dir,
             viewDir,
             anisotropicSpecular,
             at, ab,
             eta_i, eta_t,
             Ng, Ns,
             anisotropicT, anisotropicB,
             pdfSpec,
             pdfTrans);

  float pdfClear = 0.f;
  float ceta_i = entering ? 1.0f : 1.5f;
  float ceta_t = entering ? 1.5f : 1.0f;
  pdfGGXVNDF(result.dir,
             viewDir,
             clearcoatAlpha,
             ceta_i,
             ceta_t,
             Ng, Ns,
             pdfClear);

  result.pdf = pDiff*pdfDiff + pSpec*pdfSpec + pTrans*pdfTrans + pClear*pdfClear;

  result.f = evalPhysicallyBasedMaterial(mat,
                                         onDevice,
                                         attribs,
                                         objPos,
                                         primID,
                                         Ng, Ns,
                                         T, B,
                                         viewDir, result.dir);

  result.cosT = fmaxf(0.f,dot(Ns,result.dir));

  return result;
}

VSNRAY_FUNC
inline BSDFSample sampleMaterial(const dco::Material &mat,
                                 const DeviceObjectRegistry &onDevice,
                                 const dco::AttributeRec &attribs,
                                 vec3f objPos,
                                 unsigned primID,
                                 const vec3 Ng, const vec3 Ns,
                                 const vec3 T, const vec3 B,
                                 const vec3 viewDir,
                                 Random &rnd)
{
  BSDFSample result;
  if (mat.type == dco::Material::Matte) {
    auto w = faceforward(Ns, viewDir, Ng);
    auto v = fabsf(w.x) > fabsf(w.y) ? normalize(vec3(-w.z,0.f,w.x))
                                     : normalize(vec3(0.f,w.z,-w.y));
    auto u = cross(v,w);
    auto sp = cosine_sample_hemisphere(rnd(), rnd());
    result.dir = normalize(sp.x*u+sp.y*v+sp.z*w);
    result.pdf = fmaxf(0.f,dot(Ns,result.dir)) * constants::inv_pi<float>();
    result.f = evalMatteMaterial(mat,
                                 onDevice,
                                 attribs,
                                 objPos,
                                 primID,
                                 Ng, Ns,
                                 T, B,
                                 viewDir,
                                 result.dir);
    result.cosT = fmaxf(0.f,dot(Ns,result.dir));
  } else if (mat.type == dco::Material::PhysicallyBased) {
    result = samplePhysicallyBasedMaterial(mat,
                                           onDevice,
                                           attribs,
                                           objPos,
                                           primID,
                                           Ng, Ns,
                                           T, B,
                                           viewDir,
                                           rnd);
  }

  return result;
}

//=========================================================
// Light sampling
//=========================================================

struct LightSample
{
  float3 intensity;
  float3 dir;
  float3 f;
  float pdf;
  float dist;
  float dist2;
};

VSNRAY_FUNC
inline LightSample sampleLight(const dco::Light &light, vec3f hitPos, Random &rnd)
{
  LightSample result;
  light_sample<float> ls;
  if (light.type == dco::Light::Point) {
    ls = light.asPoint.sample(hitPos, rnd);
    ls.intensity = light.asPoint.intensity(hitPos);
  } else if (light.type == dco::Light::Quad) {
    ls = light.asQuad.sample(hitPos, rnd);
    ls.intensity = light.asQuad.intensity(hitPos);
  } else if (light.type == dco::Light::Directional) {
    ls = light.asDirectional.sample(hitPos, rnd);
    ls.intensity = light.asDirectional.intensity(hitPos);
  } else if (light.type == dco::Light::Spot) {
    ls = light.asSpot.sample(hitPos, rnd);
    ls.intensity = light.asSpot.intensity(ls.dir);
  } else if (light.type == dco::Light::HDRI) {
    ls = light.asHDRI.sample(hitPos, rnd);
    ls.intensity = light.asHDRI.intensity(ls.dir);
  }

  result.intensity = ls.intensity;
  result.dir = ls.dir;
  result.pdf = ls.pdf;
  result.dist = ls.dist;

  if (light.type == dco::Light::Directional
    ||light.type == dco::Light::HDRI) {
    result.dist2 = 1.f; // infinite lights are not attenuated by distance!
  } else {
    result.dist2 = ls.dist*ls.dist;
  }

  return result;
}

VSNRAY_FUNC
inline Ray clipRay(Ray ray, const float4 *clipPlanes, unsigned numClipPlanes)
{
  for (unsigned i=0; i<numClipPlanes; ++i) {
    float3 N(clipPlanes[i].xyz());
    float D(clipPlanes[i].w);
    float s = dot(N,ray.dir);
    if (s != 0.f) {
      float t = (D-dot(N,ray.ori))/s;
      if (s < 0.f) ray.tmin = fmaxf(ray.tmin,t);
      else         ray.tmax = fminf(ray.tmax,t);
    }
  }
  return ray;
}

struct HitRecLight
{
  bool hit{false};
  float t{FLT_MAX};
  bool lightVisible{false};
  unsigned lightID{UINT_MAX};
};

struct HitRec
{
  hit_record<Ray, primitive<unsigned>> surface;
  dco::HitRecordVolume volume;
  HitRecLight light;
  bool hit{false};
  bool volumeHit{false};
  bool lightHit{false};
};

template <bool EvalOpacity>
VSNRAY_FUNC
inline hit_record<Ray, primitive<unsigned>> intersectSurfaces(
    ScreenSample &ss, Ray ray,
    const DeviceObjectRegistry &onDevice,
    unsigned worldID,
    bool shadow)
{
  auto hr = intersectSurfaces(ray, onDevice.TLSs[worldID], shadow);
  while (EvalOpacity) {
    if (!hr.hit) break;

    float2 uv{hr.u, hr.v};
    const dco::Instance &inst = onDevice.instances[hr.inst_id];
    const dco::Group &group = onDevice.groups[inst.groupID];
    const dco::Geometry &geom = onDevice.geometries[group.geoms[hr.geom_id]];
    const dco::Material &mat = onDevice.materials[group.materials[hr.geom_id]];

    dco::AttributeRec attribs = getAttributes(geom,
                                              inst,
                                              float3{}, // TODO: worldPos
                                              float3{}, // TODO: worldNormal
                                              float3{}, // TODO: objectNormal
                                              float3{}, // TODO: objectNormal
                                              hr.prim_id,
                                              uv);

    float opacity
        = getOpacity(mat, onDevice, attribs, hr.isect_pos, hr.prim_id);

    float r = ss.random();
    if (r > opacity) {
      const float3 hitPos = ray.ori + hr.t * ray.dir;
      const float eps = epsilonFrom(hitPos, ray.dir, hr.t);
      ray.tmin = hr.t + eps;
      hr = intersectSurfaces(ray, onDevice.TLSs[worldID], shadow);
    } else {
      break;
    }
  }
  return hr;
}

VSNRAY_FUNC
inline dco::HitRecordVolume sampleFreeFlightDistanceAllVolumes(
    ScreenSample &ss, Ray ray, unsigned worldID,
    DeviceObjectRegistry onDevice) {

  ray.prd = &ss.random;
  return intersectVolumes(ray, onDevice.TLSs[worldID]);
}


VSNRAY_FUNC
inline dco::Light getLight(const dco::LightRef *lightRefs, unsigned lightID,
    const DeviceObjectRegistry &onDevice)
{
  mat4 xfm = mat4::identity();
  if (dco::validHandle(lightRefs[lightID].instID))
    xfm = onDevice.instances[lightRefs[lightID].instID].xfms[0];
  return xfmLight(onDevice.lights[lightRefs[lightID].lightID], xfm);
}

VSNRAY_FUNC
inline HitRecLight intersectLights(ScreenSample &ss, const Ray &ray, unsigned worldID,
    const DeviceObjectRegistry &onDevice, unsigned bounceID)
{
  HitRecLight hr;
  dco::World world = onDevice.worlds[worldID];
  for (unsigned lightID=0; lightID<world.numLights; ++lightID) {
    const dco::Light &light = getLight(world.allLights, lightID, onDevice);
    if (bounceID == 0 && !light.visible) continue;
    if (light.type == dco::Light::HDRI) {
      HitRecLight hrl = {};
      hrl.hit = true;
      hrl.t = FLT_MAX;
      hrl.lightVisible = light.visible;
      if (hrl.hit && hrl.t < hr.t
          || !hr.hit
          // if we have more than one HDRI, we have to pick an arbitrary
          // one but prefer visible over invisible lights:
          || (hrl.t == hr.t &&  (light.visible && !hr.lightVisible))) {
        hr.hit = true;
        hr.lightVisible = light.visible;
        hr.t = FLT_MAX;
        hr.lightID = lightID;
      }
    } else if (light.type == dco::Light::Quad) {
      auto hrl = intersect(ray, light.asQuad.geometry());
      float3 Nl = get_normal(hrl,light.asQuad.geometry());
      if (light.asQuad.side == dco::Light::Front && dot(Nl,ray.dir) > 0.f) continue;
      if (light.asQuad.side == dco::Light::Back && dot(Nl,ray.dir) < 0.f) continue;
      if (hrl.hit && hrl.t < hr.t) {
        hr.hit = true;
        hr.t = hrl.t;
        hr.lightVisible = light.visible;
        hr.lightID = lightID;
      }
    }
  }
  return hr;
}

VSNRAY_FUNC
inline HitRec intersectAll(ScreenSample &ss, const Ray &ray, unsigned worldID,
    const DeviceObjectRegistry &onDevice, unsigned bounceID, bool shadow)
{
  HitRec hr;
  hr.surface = intersectSurfaces<1>(ss, ray, onDevice, worldID, shadow);
  hr.light   = intersectLights(ss, ray, worldID, onDevice, bounceID/*, shadow*/);
  hr.volume  = sampleFreeFlightDistanceAllVolumes(ss, ray, worldID, onDevice/*, shadow*/);
  hr.hit = hr.surface.hit || hr.volume.hit || hr.light.hit;
  // light-hit takes precedence over surface and volume (<=)
  hr.lightHit = hr.light.hit && (!hr.surface.hit || hr.light.t <= hr.surface.t)
                             && (!hr.volume.hit || hr.light.t <= hr.volume.t);
  hr.volumeHit = hr.volume.hit && (!hr.surface.hit || hr.volume.t < hr.surface.t)
                               && (!hr.light.hit || hr.volume.t < hr.light.t);
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
  if (t < .25f) return lerp_r(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
  if (t < .5f)  return lerp_r(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
  if (t < .75f) return lerp_r(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
  if (t < 1.f)  return lerp_r(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
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
  printf("ray: [ori: (%f,%f,%f), dir: (%f,%f,%f), tmin: %f, tmax: %f, mask: %u]\n",
      ray.ori.x, ray.ori.y, ray.ori.z, ray.dir.x, ray.dir.y, ray.dir.z,
      ray.tmin, ray.tmax, ray.intersectionMask);
}

VSNRAY_FUNC
inline void printExact(const Ray &ray)
{
  unsigned orix = *(unsigned *)&ray.ori.x;
  unsigned oriy = *(unsigned *)&ray.ori.y;
  unsigned oriz = *(unsigned *)&ray.ori.z;
  unsigned dirx = *(unsigned *)&ray.dir.x;
  unsigned diry = *(unsigned *)&ray.dir.y;
  unsigned dirz = *(unsigned *)&ray.dir.z;
  unsigned tmin = *(unsigned *)&ray.tmin;
  unsigned tmax = *(unsigned *)&ray.tmax;
  printf("ray: [ori: (%x,%x,%x), dir: (%x,%x,%x), tmin: %x, tmax: %x, mask: %x]\n",
      orix, oriy, oriz, dirx, diry, dirz, tmin, tmax, ray.intersectionMask);
}
} // visionaray
