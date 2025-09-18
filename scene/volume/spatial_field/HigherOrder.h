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
  box1f valueRange;

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
    valueRange.invalidate();
    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
        for (int k=0; k<n; ++k) {
          bounds.insert(cp[i][j][k].xyz());
          valueRange.extend(cp[i][j][k].w);
        }
      }
    }
  }

  template <int S=2>
  VSNRAY_FUNC void subdivide(BezierHex *splits, int axis) const {
    assert(axis==0 || axis==1 || axis==2);

    for (int s=0; s<S; ++s) {
      splits[s].actualN = actualN;
    }

    for (int i=0; i<actualN; ++i) {
      for (int j=0; j<actualN; ++j) {
        float4 dst[N*S-(S-1)];
        float4 tmp[N*S-(S-1)];

        for (int k=0; k<actualN; ++k) {
          if (axis == 0) {
            dst[k] = cp[i][j][k];
          } else if (axis == 1) {
            dst[k] = cp[i][k][j];
          } else if (axis == 2) {
            dst[k] = cp[k][i][j];
          } else {
            assert(0);
          }
        }

        for (int n=actualN; n<actualN*S-(S-1); ++n) {
          memcpy(tmp,dst,sizeof(dst));
          int ptr=0;
          dst[ptr++] = tmp[0];
          for (int k=0; k<n-1; ++k) {
            float4 v1 = tmp[k];
            float4 v2 = tmp[k+1];

            constexpr float t{0.5f};//{1.f/S}; // TODO
            dst[ptr++] = v1*t+v2*(1-t); // TODO
          }
          dst[ptr++] = tmp[n-1];
        }

        for (int k=0; k<actualN; ++k) {
          for (int s=0; s<S; ++s) {
            if (axis == 0) {
              splits[s].cp[i][j][k] = dst[(actualN-1)*s+k];
            } else if (axis == 1) {
              splits[s].cp[i][k][j] = dst[(actualN-1)*s+k];
            } else if (axis == 2) {
              splits[s].cp[k][i][j] = dst[(actualN-1)*s+k];
            }
          }
        }
      }
    }

    for (int s=0; s<S; ++s) {
      splits[s].bounds.invalidate();
      splits[s].valueRange.invalidate();

      for (int i=0; i<actualN; ++i) {
        for (int j=0; j<actualN; ++j) {
          for (int k=0; k<actualN; ++k) {
            splits[s].bounds.insert(splits[s].cp[i][j][k].xyz());
            splits[s].valueRange.extend(splits[s].cp[i][j][k].w);
          }
        }
      }
    }
  }

  template <int S>
  VSNRAY_FUNC void subdivide(BezierHex *splits) {
    int num=0;
    splits[num++] = *this;
    while (num < S) {
      auto hex = splits[0];
      int splitAxis = hex.bestSplitAxis();
      BezierHex childs[2];
      subdivide<2>(childs,splitAxis);
      splits[0] = childs[0];
      splits[num++] = childs[1];
      std::sort(splits,splits+num,
        [](BezierHex a, BezierHex b)
        {
          return volume(a.bounds) > volume(b.bounds);
        });
    }
  }

  VSNRAY_FUNC
  float value() const {
  #if 1
    return (valueRange.min+valueRange.max)*0.5f;
  #else
    int n=actualN;
    float v1 = (cp[0][0][0].w+cp[n-1][0][0].w)*0.5f;
    float v2 = (cp[0][n-1][0].w+cp[n-1][n-1][0].w)*0.5f;
    float v3 = (cp[0][0][n-1].w+cp[n-1][0][n-1].w)*0.5f;
    float v4 = (cp[0][n-1][n-1].w+cp[n-1][n-1][n-1].w)*0.5f;

    float v5 = (v1+v3)*0.5f;
    float v6 = (v2+v4)*0.5f;

    return (v5+v6)*0.5f;
  #endif
  }

  VSNRAY_FUNC
  int bestSplitAxis() const {
    //return max_index(bounds.size());
    float3 minV(FLT_MAX);
    for (int i=0; i<std::min(actualN-1,MAX_ELEM_ORDER+0); ++i) {
      for (int j=0; j<std::min(actualN-1,MAX_ELEM_ORDER+0); ++j) {
        for (int k=0; k<std::min(actualN-1,MAX_ELEM_ORDER+0); ++k) {
          minV.x = fminf(minV.x,length(cp[i][j][k].xyz()-cp[i][j][k+1].xyz()));
          minV.y = fminf(minV.y,length(cp[i][j][k].xyz()-cp[i][j+1][k].xyz()));
          minV.z = fminf(minV.z,length(cp[i][j][k].xyz()-cp[i+1][j][k].xyz()));
        }
      }
    }
    return max_index(minV);
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
  for (int i=0; i<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++i) {
    for (int j=0; j<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++j) {
      for (int k=0; k<std::min(hex.actualN,MAX_ELEM_ORDER+1); ++k) {
        out << hex.cp[i][j][k];
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

  const float threshold = volume(hex.bounds) * 1e-5f;

next:
  while (ptr > 0 && ptr < STACK_MAX) {
    hex = stack[--ptr];

    while (volume(hex.bounds) > threshold && ptr < STACK_MAX) {
      int splitAxis = hex.bestSplitAxis();

      constexpr int S=2; // num splits
      uelem::BezierHex splits[S];
      hex.subdivide<S>(splits,splitAxis);

      bool b[S];
      for (int s=0; s<S; ++s) {
        b[s] = splits[s].bounds.contains(pos);
      }

      bool assigned = false;
      for (int s=0; s<S; ++s) {
        if (!b[s]) continue;
        if (assigned) {
          stack[ptr++] = splits[s];
        } else {
          hex = splits[s];
          assigned = true;
        }
      }

      if (!assigned) {
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
