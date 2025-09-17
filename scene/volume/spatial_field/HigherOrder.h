// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace visionaray {

#define MAX_ELEM_ORDER 5

namespace uelem {

struct BezierHex
{
  constexpr static int N = MAX_ELEM_ORDER+1;
  float4 cp[N][N][N];
  int actualN;
  aabb bounds;

  BezierHex() = default;

  VSNRAY_FUNC
  BezierHex(const float4 *vertices, const uint64_t *indices, int order)
    : actualN(order+1)
  {
    int n=actualN;
    auto v = [&](int i) { return vertices[indices[i]]; };

    // corner points:
    cp[0][0][0] = v(0);
    cp[n-1][0][0] = v(1);
    cp[n-1][n-1][0] = v(2);
    cp[0][n-1][0] = v(3);
    cp[0][0][n-1] = v(4);
    cp[n-1][0][n-1] = v(5);
    cp[n-1][n-1][n-1] = v(6);
    cp[0][n-1][n-1] = v(7);

    // edge points:
    int e1 = 8;
    int e2 = e1+n-2;
    int e3 = e2+n-2;
    int e4 = e3+n-2;
    int e5 = e4+n-2;
    int e6 = e5+n-2;
    int e7 = e6+n-2;
    int e8 = e7+n-2;
    int e9 = e8+n-2;
    int e10 = e9+n-2;
    int e11 = e10+n-2;
    int e12 = e11+n-2;

    for (int i=0; i<n-2; ++i) {
      cp[i+1][0][0] = v(e1+i);
      cp[n-1][i+1][0] = v(e2+i);
      cp[i+1][n-1][0] = v(e3+i);
      cp[0][i+1][0] = v(e4+i);

      cp[i+1][0][n-1] = v(e5+i);
      cp[n-1][i+1][n-1] = v(e6+i);
      cp[i+1][n-1][n-1] = v(e7+i);
      cp[0][i+1][n-1] = v(e8+i);

      cp[0][0][i+1] = v(e9+i);
      cp[n-1][0][i+1] = v(e10+i);
      cp[n-1][n-1][i+1] = v(e11+i);
      cp[0][n-1][i+1] = v(e12+i);
    }

    // face points:
    int f1 = e12+n-2;
    int f2 = f1+(n-2)*(n-2);
    int f3 = f2+(n-2)*(n-2);
    int f4 = f3+(n-2)*(n-2);
    int f5 = f4+(n-2)*(n-2);
    int f6 = f5+(n-2)*(n-2);

    for (int i=0; i<n-2; ++i) {
      for (int j=0; j<n-2; ++j) {
        cp[j+1][i+1][0] = v(f5+i*(n-2)+j);
        cp[0][j+1][i+1] = v(f1+i*(n-2)+j);
        cp[j+1][0][i+1] = v(f3+i*(n-2)+j);

        cp[j+1][i+1][n-1] = v(f6+i*(n-2)+j);
        cp[n-1][j+1][i+1] = v(f2+i*(n-2)+j);
        cp[j+1][n-1][i+1] = v(f4+i*(n-2)+j);
      }
    }

    // volume points:
    int v1 = f6+(n-2)*(n-2);
    for (int i=0; i<n-2; ++i) {
      for (int j=0; j<n-2; ++j) {
        for (int k=0; k<n-2; ++k) {
          // TODO: are those row-major or z-order?!
          cp[k+1][j+1][i+1] = v(v1+i*(n-2)*(n-2)+j*(n-2)+k);
        }
      }
    }

    flipXZ();

    bounds.invalidate();
    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
        for (int k=0; k<n; ++k) {
          bounds.insert(cp[i][j][k].xyz());
        }
      }
    }
  }

  VSNRAY_FUNC
  void subdivide(BezierHex &a, BezierHex &b, int axis) {
    assert(axis==0 || axis==1 || axis==2);

    a.actualN = actualN;
    b.actualN = actualN;

    a.bounds.invalidate();
    b.bounds.invalidate();

    if (axis == 1) {
      flipXY();
    }

    if (axis == 2) {
      flipXZ();
    }

    for (int i=0; i<actualN; ++i) {
      for (int j=0; j<actualN; ++j) {
        float4 dst[N*2-1];
        float4 tmp[N*2-1];

        memcpy(dst,&cp[i][j],sizeof(cp[i][j]));
        for (int n=actualN; n<actualN*2-1; ++n) {
          memcpy(tmp,dst,sizeof(dst));
          int ptr=0;
          dst[ptr++] = tmp[0];
          for (int k=0; k<n-1; ++k) {
            float4 v1 = tmp[k];
            float4 v2 = tmp[k+1];
            dst[ptr++] = v1*0.5f+v2*0.5f;
          }
          dst[ptr++] = tmp[n-1];
        }

        for (int k=0; k<actualN; ++k) {
          a.cp[i][j][k] = dst[k];
          b.cp[i][j][k] = dst[actualN-1+k];

          a.bounds.insert(a.cp[i][j][k].xyz());
          b.bounds.insert(b.cp[i][j][k].xyz());
        }
      }
    }

    if (axis == 1) {
      a.flipXY();
      b.flipXY();
      flipXY();
    }

    if (axis == 2) {
      a.flipXZ();
      b.flipXZ();
      flipXZ();
    }
  }

  VSNRAY_FUNC
  float value() const {
    int n=actualN;
    float v1 = (cp[0][0][0].w+cp[n-1][0][0].w)*0.5f;
    float v2 = (cp[0][n-1][0].w+cp[n-1][n-1][0].w)*0.5f;
    float v3 = (cp[0][0][n-1].w+cp[n-1][0][n-1].w)*0.5f;
    float v4 = (cp[0][n-1][n-1].w+cp[n-1][n-1][n-1].w)*0.5f;

    float v5 = (v1+v3)*0.5f;
    float v6 = (v2+v4)*0.5f;

    return (v5+v6)*0.5f;
  }

  VSNRAY_FUNC
  inline void flipXY() {
    for (int i=0; i<std::min(actualN,MAX_ELEM_ORDER+1); ++i) {
      for (int j=0; j<std::min(actualN,MAX_ELEM_ORDER+1); ++j) {
        for (int k=j+1; k<std::min(actualN,MAX_ELEM_ORDER+1); ++k) {
          auto tmp = cp[i][j][k];
          cp[i][j][k] = cp[i][k][j];
          cp[i][k][j] = tmp;
        }
      }
    }
  }

  VSNRAY_FUNC
  inline void flipXZ() {
    for (int i=0; i<std::min(actualN,MAX_ELEM_ORDER+1); ++i) {
      for (int j=0; j<std::min(actualN,MAX_ELEM_ORDER+1); ++j) {
        for (int k=i+1; k<std::min(actualN,MAX_ELEM_ORDER+1); ++k) {
          auto tmp = cp[i][j][k];
          cp[i][j][k] = cp[k][j][i];
          cp[k][j][i] = tmp;
        }
      }
    }
  }
};

inline
std::ostream &operator<<(std::ostream &out, BezierHex hex) {
  for (int z=0; z<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++z) {
    for (int y=0; y<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++y) {
      for (int x=0; x<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++x) {
        out << hex.cp[x][y][z];
      }
      out << '\n';
    }
    out << '\n';
  }
  return out;
}

} // namespace uelem

VSNRAY_FUNC
inline bool intersectBezierHex(float &value, float3 pos,
                               const float4 *vertices,
                               const uint64_t *indices,
                               int numVerts)
{
  int D = cbrt(numVerts) - 1;
  assert(D <= MAX_ELEM_ORDER && "Element order out of range!");

  uelem::BezierHex hex(vertices, indices, D);

  #define STACK_MAX 32
  uelem::BezierHex stack[STACK_MAX];
  int ptr = 0;
  stack[ptr++] = hex;

  bool contained = false;

  const float threshold = volume(hex.bounds) * 0.00001f;

next:
  while (ptr > 0 && ptr < STACK_MAX) {
    hex = stack[--ptr];

    while (volume(hex.bounds) > threshold && ptr < STACK_MAX-1) {
      int splitAxis = max_index(hex.bounds.size());

      uelem::BezierHex a, b;
      hex.subdivide(a,b,splitAxis);

      bool b1 = a.bounds.contains(pos);
      bool b2 = b.bounds.contains(pos);

      if (b1 && b2) {
        hex = a;
        stack[ptr++] = b;
      } else if (b1) {
        hex = a;
      } else if (b2) {
        hex = b;
      } else {
        goto next;
      }
    }

    auto leafBounds = hex.bounds;
    if (volume(leafBounds) < threshold) {
      if (leafBounds.contains(pos)) {
        value = hex.value();
        return true;
      }
    }
  }

  return false;
}

} // namespace visionaray
