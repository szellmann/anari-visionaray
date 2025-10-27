// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace visionaray {

namespace dco {

#define ELEM_ORDER 3

struct BezierHex;
std::ostream &operator<<(std::ostream &out, BezierHex hex);

struct BezierHex
{
  constexpr static int N = ELEM_ORDER+1;

  unsigned elemID;
  float4 cp[N*N*N];
  box4f bounds;

  BezierHex() = default;

  VSNRAY_FUNC void deCasteljauSubdiv(
      BezierHex &child0, BezierHex &child1, float t, int axis, int i, int j) const
  {
    float4 tmp[(N*(N+1))/2];

    for (int k=0; k<N; ++k) {
      if (axis == 0) {
        tmp[k] = get(i,j,k);
      } else if (axis == 1) {
        tmp[k] = get(i,k,j);
      } else if (axis == 2) {
        tmp[k] = get(k,i,j);
      }
    }

    float4 *src = tmp;
    float4 *dst = tmp + N;

    for (int k=1; k<N; ++k) {
      for (int l=0; l<N-k; ++l) {
        dst[l] = (1.f-t)*src[l] + t*src[l+1];
      }
      src += N-k+1;
      dst += N-k;
    }

    src = tmp;
    for (int k=0; k<N; ++k) {
      if (axis == 0) {
        child0(i,j,k) = src[0];
        child1(i,j,N-k-1) = src[N-k-1];
      } else if (axis == 1) {
        child0(i,k,j) = src[0];
        child1(i,N-k-1,j) = src[N-k-1];
      } else if (axis == 2) {
        child0(k,i,j) = src[0];
        child1(N-k-1,i,j) = src[N-k-1];
      }
      src += N-k;
    }
  }

  VSNRAY_FUNC void deCasteljauSubdiv_N4(
      BezierHex &child0, BezierHex &child1, float t, int axis, int i, int j) const
  {
    float4 v00, v01, v02, v03;
    if (axis == 0) {
      v00 = get(i,j,0);
      v01 = get(i,j,1);
      v02 = get(i,j,2);
      v03 = get(i,j,3);
    } else if (axis == 1) {
      v00 = get(i,0,j);
      v01 = get(i,1,j);
      v02 = get(i,2,j);
      v03 = get(i,3,j);
    } else if (axis == 2) {
      v00 = get(0,i,j);
      v01 = get(1,i,j);
      v02 = get(2,i,j);
      v03 = get(3,i,j);
    }

    float4 v10 = (1-t)*v00 + t*v01;
    float4 v11 = (1-t)*v01 + t*v02;
    float4 v12 = (1-t)*v02 + t*v03;

    float4 v20 = (1-t)*v10 + t*v11;
    float4 v21 = (1-t)*v11 + t*v12;

    float4 v30 = (1-t)*v20 + t*v21;

    if (axis == 0) {
      child0(i,j,0) = v00;
      child0(i,j,1) = v10;
      child0(i,j,2) = v20;
      child0(i,j,3) = v30;

      child1(i,j,0) = v30;
      child1(i,j,1) = v21;
      child1(i,j,2) = v12;
      child1(i,j,3) = v03;
    } else if (axis == 1) {
      child0(i,0,j) = v00;
      child0(i,1,j) = v10;
      child0(i,2,j) = v20;
      child0(i,3,j) = v30;

      child1(i,0,j) = v30;
      child1(i,1,j) = v21;
      child1(i,2,j) = v12;
      child1(i,3,j) = v03;
    } else if (axis == 2) {
      child0(0,i,j) = v00;
      child0(1,i,j) = v10;
      child0(2,i,j) = v20;
      child0(3,i,j) = v30;

      child1(0,i,j) = v30;
      child1(1,i,j) = v21;
      child1(2,i,j) = v12;
      child1(3,i,j) = v03;
    }
  }

  VSNRAY_FUNC void split2(BezierHex &child0, BezierHex &child1, int axis) const {
    assert(axis==0 || axis==1 || axis==2);

    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        if constexpr (N == 4) {
          deCasteljauSubdiv_N4(child0,child1,0.5f,axis,i,j);
        } else {
        //  deCasteljauSubdiv(child0,child1,0.5f,axis,i,j);
        }
      }
    }

    child0.recomputeBounds();
    child1.recomputeBounds();
  }

  VSNRAY_FUNC void split4(BezierHex &child0, BezierHex &child1,
                          BezierHex &child2, BezierHex &child3, int axis) const {
    assert(axis==0 || axis==1 || axis==2);

    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        if constexpr (N == 4) {
          deCasteljauSubdiv_N4(child0,child1,0.25f,axis,i,j);
          child1.deCasteljauSubdiv_N4(child1,child2,1/3.f,axis,i,j);
          child2.deCasteljauSubdiv_N4(child2,child3,0.5f,axis,i,j);
        } else {
        //  deCasteljauSubdiv(splits,t,axis,i,j);
        }
      }
    }

    child0.recomputeBounds();
    child1.recomputeBounds();
    child2.recomputeBounds();
    child3.recomputeBounds();
  }

  VSNRAY_FUNC void recomputeBounds() {
    bounds.min = vec4f(FLT_MAX);
    bounds.max = vec4f(-FLT_MAX);

    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        for (int k=0; k<N; ++k) {
          bounds.extend(get(i,j,k));
        }
      }
    }
  }

  VSNRAY_FUNC BezierHex clip(const box3f &uvw, BezierHex &child0,
                             BezierHex &child1, BezierHex &child2) const {
    // x
    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        if constexpr (N == 4) {
          deCasteljauSubdiv_N4(child0,child1,uvw.max[0],0,i,j);
          child0.deCasteljauSubdiv_N4(child1,child2,uvw.min[0]/uvw.max[0],0,i,j);
        } else {
        //  deCasteljauSubdiv(children,t,axis,i,j);
        }
      }
    }
    // y
    BezierHex hex = child2;
    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        if constexpr (N == 4) {
          hex.deCasteljauSubdiv_N4(child0,child1,uvw.max[1],1,i,j);
          child0.deCasteljauSubdiv_N4(child1,child2,uvw.min[1]/uvw.max[1],1,i,j);
        } else {
        //  deCasteljauSubdiv(children,t,axis,i,j);
        }
      }
    }
    // z
    hex = child2;
    for (int i=0; i<N; ++i) {
      for (int j=0; j<N; ++j) {
        if constexpr (N == 4) {
          hex.deCasteljauSubdiv_N4(child0,child1,uvw.max[2],2,i,j);
          child0.deCasteljauSubdiv_N4(child1,child2,uvw.min[2]/uvw.max[2],2,i,j);
        } else {
        //  deCasteljauSubdiv(children,t,axis,i,j);
        }
      }
    }
    child2.recomputeBounds();
    return child2;
  }

  template <int S>
  VSNRAY_FUNC void subdivide(BezierHex *splits) {
    int num=0;
    splits[num++] = *this;
    while (num < S) {
      auto hex = splits[0];
      int splitAxis = hex.bestSplitAxis();
      BezierHex childs[2];
      split2(childs[0],childs[1],splitAxis);
      splits[0] = childs[0];
      splits[num++] = childs[1];
      std::sort(splits,splits+num,
        [](BezierHex a, BezierHex b)
        {
          return volume(a.spatialBounds()) > volume(b.spatialBounds());
        });
    }
  }

  VSNRAY_FUNC
  float4 &operator()(int i, int j, int k) {
    return cp[i*N*N+j*N+k];
  }

  VSNRAY_FUNC
  const float4 &operator()(int i, int j, int k) const {
    return cp[i*N*N+j*N+k];
  }

  VSNRAY_FUNC
  float4 &get(int i, int j, int k) {
    return cp[i*N*N+j*N+k];
  }

  VSNRAY_FUNC
  const float4 &get(int i, int j, int k) const {
    return cp[i*N*N+j*N+k];
  }

  VSNRAY_FUNC
  float value() const {
  #if 1
    return (bounds.min.w+bounds.max.w)*0.5f;
  #else
    float v1 = (get(0,0,0).w+get(N-1,0,0).w)*0.5f;
    float v2 = (get(0,N-1,0).w+get(N-1,N-1,0).w)*0.5f;
    float v3 = (get(0,0,N-1).w+get(N-1,0,N-1).w)*0.5f;
    float v4 = (get(0,N-1,N-1).w+get(N-1,N-1,N-1).w)*0.5f;

    float v5 = (v1+v3)*0.5f;
    float v6 = (v2+v4)*0.5f;

    return (v5+v6)*0.5f;
  #endif
  }

  VSNRAY_FUNC
  aabb spatialBounds() const {
    return aabb(bounds.min.xyz(),bounds.max.xyz());
  }

  VSNRAY_FUNC
  box1f valueRange() const {
    return {bounds.min.w,bounds.max.w};
  }

  VSNRAY_FUNC
  int bestSplitAxis() const {
    //return max_index(spatialBounds().size());
    float3 minV(FLT_MAX);
    for (int i=0; i<N-1; ++i) {
      for (int j=0; j<N-1; ++j) {
        for (int k=0; k<N-1; ++k) {
          minV.x = fminf(minV.x,length(get(i,j,k).xyz()-get(i,j,k+1).xyz()));
          minV.y = fminf(minV.y,length(get(i,j,k).xyz()-get(i,j+1,k).xyz()));
          minV.z = fminf(minV.z,length(get(i,j,k).xyz()-get(i+1,j,k).xyz()));
        }
      }
    }
    return max_index(minV);
  }

  // Helper function to swizzle from VTU layout to ours
  VSNRAY_FUNC
  static BezierHex fromVTU(const float4 *vertices, const uint64_t *indices)
  {
    auto v = [&](int i) { return vertices[indices[i]]; };
  
    constexpr int N = BezierHex::N;
  
    BezierHex hex;
  
    // corner points:
    hex(0,0,0) = v(0);
    hex(0,0,N-1) = v(1);
    hex(0,N-1,N-1) = v(2);
    hex(0,N-1,0) = v(3);
    hex(N-1,0,0) = v(4);
    hex(N-1,0,N-1) = v(5);
    hex(N-1,N-1,N-1) = v(6);
    hex(N-1,N-1,0) = v(7);
  
    // edge points:
    int e1 = 8;
    int e2 = e1+N-2;
    int e3 = e2+N-2;
    int e4 = e3+N-2;
    int e5 = e4+N-2;
    int e6 = e5+N-2;
    int e7 = e6+N-2;
    int e8 = e7+N-2;
    int e9 = e8+N-2;
    int e10 = e9+N-2;
    int e11 = e10+N-2;
    int e12 = e11+N-2;
  
    for (int i=0; i<N-2; ++i) {
      hex(0,0,i+1) = v(e1+i);
      hex(0,i+1,N-1) = v(e2+i);
      hex(0,N-1,i+1) = v(e3+i);
      hex(0,i+1,0) = v(e4+i);
  
      hex(N-1,0,i+1) = v(e5+i);
      hex(N-1,i+1,N-1) = v(e6+i);
      hex(N-1,N-1,i+1) = v(e7+i);
      hex(N-1,i+1,0) = v(e8+i);
  
      hex(i+1,0,0) = v(e9+i);
      hex(i+1,0,N-1) = v(e10+i);
      hex(i+1,N-1,N-1) = v(e11+i);
      hex(i+1,N-1,0) = v(e12+i);
    }
  
    // face points:
    int f1 = e12+N-2;
    int f2 = f1+(N-2)*(N-2);
    int f3 = f2+(N-2)*(N-2);
    int f4 = f3+(N-2)*(N-2);
    int f5 = f4+(N-2)*(N-2);
    int f6 = f5+(N-2)*(N-2);
  
    for (int i=0; i<N-2; ++i) {
      for (int j=0; j<N-2; ++j) {
        hex(0,i+1,j+1) = v(f5+i*(N-2)+j);
        hex(i+1,j+1,0) = v(f1+i*(N-2)+j);
        hex(i+1,0,j+1) = v(f3+i*(N-2)+j);
  
        hex(N-1,i+1,j+1) = v(f6+i*(N-2)+j);
        hex(i+1,j+1,N-1) = v(f2+i*(N-2)+j);
        hex(i+1,N-1,j+1) = v(f4+i*(N-2)+j);
      }
    }
  
    // volume points:
    int v1 = f6+(N-2)*(N-2);
    for (int i=0; i<N-2; ++i) {
      for (int j=0; j<N-2; ++j) {
        for (int k=0; k<N-2; ++k) {
          hex(i+1,j+1,k+1) = v(v1+i*(N-2)*(N-2)+j*(N-2)+k);
        }
      }
    }
  
    hex.recomputeBounds();
  
    return hex;
  }
};

inline
std::ostream &operator<<(std::ostream &out, BezierHex hex) {
  for (int i=0; i<BezierHex::N; ++i) {
    for (int j=0; j<BezierHex::N; ++j) {
      for (int k=0; k<BezierHex::N; ++k) {
        out << hex(i,j,k);
      }
      out << '\n';
    }
    out << '\n';
  }
  return out;
}

VSNRAY_FUNC
inline aabb get_bounds(const BezierHex &hex)
{
  return hex.spatialBounds();
}

inline void split_primitive(
    aabb &L, aabb &R, float plane, int axis, const BezierHex &hex)
{
  assert(0);
}

VSNRAY_FUNC inline float volume(box3f uvw) {
  return volume(aabb(uvw.min,uvw.max));
}

VSNRAY_FUNC
inline bool intersectBezierHex(const dco::BezierHex &hex, const float3 P, float &retVal)
{
  #define TRAVERSAL_WIDTH 2
  //#define TRAVERSAL_WIDTH 4

  #define SCRATCH_SIZE TRAVERSAL_WIDTH >= 3 ? TRAVERSAL_WIDTH : 3
  dco::BezierHex scratch[SCRATCH_SIZE];

  dco::BezierHex curr = hex;
  box3f uvw({0.f,0.f,0.f},{1.f,1.f,1.f});

  #define STACK_MAX 32
  box3f stack[STACK_MAX];
  int ptr = 0;
  stack[ptr++] = uvw;

  const float threshold = volume(hex.spatialBounds()) * 1e-5f;

next:
  while (ptr > 0 && ptr < STACK_MAX) {
    uvw = stack[--ptr];

    while (volume(curr.spatialBounds()) > threshold && ptr < STACK_MAX) {
      int splitAxis = curr.bestSplitAxis();

      curr = hex.clip(uvw,scratch[0],scratch[1],scratch[2]);
  
      constexpr int S=TRAVERSAL_WIDTH;
#if TRAVERSAL_WIDTH == 4
      box3f children[4] = {uvw,uvw,uvw,uvw};
      float w = (uvw.max[splitAxis]-uvw.min[splitAxis])*0.25f;
      children[0].max[splitAxis] = uvw.min[splitAxis] + w;
      children[1].min[splitAxis] = uvw.min[splitAxis] + w;
      children[1].max[splitAxis] = uvw.min[splitAxis] + 2*w;
      children[2].min[splitAxis] = uvw.min[splitAxis] + 2*w;
      children[2].max[splitAxis] = uvw.min[splitAxis] + 3*w;
      children[3].min[splitAxis] = uvw.min[splitAxis] + 3*w;

      curr.split4(scratch[0],scratch[1],scratch[2],scratch[3],splitAxis);
#elif TRAVERSAL_WIDTH == 2
      box3f children[2] = {uvw,uvw};
      children[0].max[splitAxis] = (uvw.min[splitAxis]+uvw.max[splitAxis])*0.5f;
      children[1].min[splitAxis] = (uvw.min[splitAxis]+uvw.max[splitAxis])*0.5f;

      curr.split2(scratch[0],scratch[1],splitAxis);
#endif

      bool b[S];
      for (int s=0; s<S; ++s) {
        b[s] = scratch[s].spatialBounds().contains(P);
      }

      bool assigned = false;
      for (int s=0; s<S; ++s) {
        if (!b[s]) continue;
        if (assigned) {
          stack[ptr++] = children[s];
        } else {
          uvw = children[s];
          curr = scratch[s];
          assigned = true;
        }
      }

      if (!assigned) {
        goto next;
      }
    }

    auto leafBounds = curr.spatialBounds();
    if (volume(leafBounds) < threshold) {
      if (leafBounds.contains(P)) {
        retVal = curr.value();
        return true;
      }
    }
  }

  return false;
}

template <typename R>
VSNRAY_FUNC
inline hit_record<R, primitive<unsigned>> intersect(const R &ray, const BezierHex &hex)
{
  hit_record<R, primitive<unsigned>> result;
  float3 pos = ray.ori;
  float value = 0.f;

  bool hit = intersectBezierHex(hex, pos, value);
  result.hit = hit;

  if (result.hit) {
    result.t = 0.f;
    result.prim_id = hex.elemID;
    result.u = value; // misuse "u" to store value
  }

  return result;
}


} // namespace visionaray::dco
} // namespace visionaray
