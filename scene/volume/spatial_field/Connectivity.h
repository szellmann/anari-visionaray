// Copyright 2023-2025 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "DeviceCopyableObjects.h"

namespace visionaray {

namespace conn {

__device__
static const int Tet[4][3] {
  { 1, 3, 2 },
  { 0, 2, 3 },
  { 0, 3, 1 },
  { 0, 1, 2 },
};

__device__
static const int Pyr[5][4] {
  { 0, 1, 3, 2 },
  { 0, 4, 1, -1 },
  { 2, 4, 3, -1 },
  { 0, 3, 4, -1 },
  { 1, 4, 2, -1 },
};

__device__
static const int Wed[5][4] {
  { 0, 1, 3, 4 },
  { 0, 2, 1, -1 },
  { 3, 4, 5, -1 },
  { 0, 3, 2, 5 },
  { 1, 2, 4, 5 },
};

__device__
static const int Hex[6][4] {
  { 0, 4, 1, 5 },
  { 3, 2, 7, 6 },
  { 0, 3, 4, 7 },
  { 1, 5, 2, 6 },
  { 4, 7, 5, 6 },
  { 0, 1, 3, 2 },
};

struct UniqueFace {
  uint64_t i1;
  uint64_t i2;
  uint64_t i3;
};

struct UniqueFaceHash
{
  std::size_t operator()(const UniqueFace& f) const
  {
    return ((std::hash<uint64_t>()(f.i1)
             ^ (std::hash<uint64_t>()(f.i2) << 1)) >> 1)
             ^ (std::hash<uint64_t>()(f.i3) << 1);
  }
};

struct UniqueFaceEqual
{
  bool operator()(const UniqueFace &f1, const UniqueFace &f2) const
  {
    return f1.i1==f2.i1 && f1.i2==f2.i2 && f1.i3==f2.i3;
  }
};

// ========================================================
// element helpers
// ========================================================

struct Face
{
  Face() = default;

  __host__ __device__
  Face(const float4 *vertices, uint64_t i0, uint64_t i1, uint64_t i2)
    : vertices(vertices)
  {
    indices[0] = i0;
    indices[1] = i1;
    indices[2] = i2;
    indices[3] = ~0ull;
  }

  __host__ __device__
  Face(const float4 *vertices, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3)
    : vertices(vertices)
  {
    indices[0] = i0;
    indices[1] = i1;
    indices[2] = i2;
    indices[3] = i3;
  }

  __host__ __device__
  inline int numTriangles() const
  {
    if (indices[3] == ~0ull) return 1;
    else return 2;
  }

  __host__ __device__
  inline float4 vertex(int i) const
  {
    return vertices[indices[i]];
  }

  __host__ __device__
  inline basic_triangle<3,float> triangle(int i) const
  {
    float3 v1, v2, v3;
    if (i==0) {
      v1 = vertices[indices[0]].xyz();
      v2 = vertices[indices[1]].xyz();
      v3 = vertices[indices[2]].xyz();
    } else {
      v1 = vertices[indices[1]].xyz();
      v2 = vertices[indices[3]].xyz();
      v3 = vertices[indices[2]].xyz();
    }
    basic_triangle<3,float> tri;
    tri.v1 = v1;
    tri.e2 = v2-v1;
    tri.e1 = v3-v1;
    return tri;
  }

  __host__
  inline bool isPlanar() const
  {
    if (numTriangles() == 1)
      return true;

    auto tri0 = triangle(0);
    auto tri1 = triangle(1);

    float3 n0 = normalize(cross(tri0.e1,tri0.e2));
    float3 n1 = normalize(cross(tri1.e1,tri1.e2));

    return (1.f-fabsf(dot(n0,n1))) < 1e-3f;
  }

  const float4 *vertices;
  uint64_t indices[4];
};

struct UElem
{
  __host__ __device__
  UElem(const dco::UElem &elem)
  {
    numVertices = elem.end-elem.begin;

    if (numVertices < 4 || numVertices > 8)
      return;

    for (int i=0; i<numVertices; ++i) {
      uint64_t idx = elem.indexBuffer[elem.begin+i];
      vertices[i] = elem.vertexBuffer[idx];
      indices[i] = idx;
    }
  }

  __host__ __device__
  int numFaces() const
  {
    if (numVertices == 4) return 4;
    else if (numVertices == 5) return 5;
    else if (numVertices == 6) return 5;
    else if (numVertices == 8) return 6;
    assert(0);
    return -1;
  }

  __host__ __device__
  inline Face face(int i) const
  {
    if (numVertices == 4) {
      return Face(vertices,Tet[i][0],Tet[i][1],Tet[i][2]);
    } else if (numVertices == 5) {
      return Face(vertices,Pyr[i][0],Pyr[i][1],Pyr[i][2],Pyr[i][3]);
    } else if (numVertices == 6) {
      return Face(vertices,Wed[i][0],Wed[i][1],Wed[i][2],Wed[i][3]);
    } else if (numVertices == 8) {
      return Face(vertices,Hex[i][0],Hex[i][1],Hex[i][2],Hex[i][3]);
    }
    assert(0);
    return {};
  }

  __host__
  inline UniqueFace uniqueFace(int i) const
  {
    uint64_t I[4];
    if (numVertices == 4) {
      I[0] = indices[Tet[i][0]];
      I[1] = indices[Tet[i][1]];
      I[2] = indices[Tet[i][2]];
      std::sort(I,I+3);
    } else if (numVertices == 5) {
      if (face(i).numTriangles() == 1) {
        I[0] = indices[Pyr[i][0]];
        I[1] = indices[Pyr[i][1]];
        I[2] = indices[Pyr[i][2]];
        std::sort(I,I+3);
      } else {
        I[0] = indices[Pyr[i][0]];
        I[1] = indices[Pyr[i][1]];
        I[2] = indices[Pyr[i][2]];
        I[3] = indices[Pyr[i][3]];
        std::sort(I,I+4);
      }
    } else if (numVertices == 6) {
      if (face(i).numTriangles() == 1) {
        I[0] = indices[Wed[i][0]];
        I[1] = indices[Wed[i][1]];
        I[2] = indices[Wed[i][2]];
        std::sort(I,I+3);
      } else {
        I[0] = indices[Wed[i][0]];
        I[1] = indices[Wed[i][1]];
        I[2] = indices[Wed[i][2]];
        I[3] = indices[Wed[i][3]];
        std::sort(I,I+4);
      }
    } else if (numVertices == 8) {
      if (face(i).numTriangles() == 1) {
        I[0] = indices[Hex[i][0]];
        I[1] = indices[Hex[i][1]];
        I[2] = indices[Hex[i][2]];
        std::sort(I,I+3);
      } else {
        I[0] = indices[Hex[i][0]];
        I[1] = indices[Hex[i][1]];
        I[2] = indices[Hex[i][2]];
        I[3] = indices[Hex[i][3]];
        std::sort(I,I+4);
      }
    }

    UniqueFace f;
    f.i1 = I[0];
    f.i2 = I[1];
    f.i3 = I[2];
    return f;
  }

  __host__
  inline bool allFacesPlanar() const
  {
    for (int i=0; i<numFaces(); ++i) {
      if (!face(i).isPlanar()) return false;
    }
    return true;
  }

  __host__
  inline bool hasCoplanarFaces() const
  {
    for (int i=0; i<numFaces(); ++i) {
      const Face f1 = face(i);
      if (!f1.isPlanar()) continue;
      auto tri1 = f1.triangle(0);
      float3 n1 = normalize(cross(tri1.e1,tri1.e2));
      for (int j=i+1; j<numFaces(); ++j) {
        const Face f2 = face(j);
        if (!f2.isPlanar()) continue;
        auto tri2 = f2.triangle(0);
        float3 n2 = normalize(cross(tri2.e1,tri2.e2));
        float3 diff = abs(n1-n2);
        const float eps = 1e-4f;
        if (diff.x<eps && diff.y<eps && diff.z<eps) {
          return true;
        }
      }
    }
    return false;
  }

  // test if *either* of the faces' winding order is wrong
  __host__
  inline bool checkWindingOrder() const
  {
    for (int i=0; i<numFaces(); ++i) {
      const Face f = face(i);
      auto tri = f.triangle(0);
      const Plane p = makePlane(tri.v1,tri.e1+tri.v1,tri.e2+tri.v1);

      const float eps = 1e-4f;

      // find a vertex that is not in this plane:
      float sgn = 0.f;
      for (int j=0; j<numVertices; ++j) {
        float3 refp = vertices[j].xyz();
        sgn = p.eval(refp);
        if (fabsf(sgn) > eps) break;
      }
      if (sgn > eps) return false;
    }

    // all normals facing inwards: check passed
    return true;
  }

  // test if *all* of the faces' winding order is wrong
  __host__
  inline bool checkWindingOrderFlipped() const
  {
    for (int i=0; i<numFaces(); ++i) {
      const Face f = face(i);
      auto tri = f.triangle(0);
      const Plane p = makePlane(tri.v1,tri.e1+tri.v1,tri.e2+tri.v1);

      const float eps = 1e-4f;

      // find a vertex that is not in this plane:
      float sgn = 0.f;
      for (int j=0; j<numVertices; ++j) {
        float3 refp = vertices[j].xyz();
        sgn = p.eval(refp);
        if (fabsf(sgn) > eps) break;
      }
      if (sgn < -eps) return false;
    }

    // all normals facing outwards: this element could be flipped!
    return true;
  }

  __host__
  inline bool isValid() const
  {
    // TODO: test and enable winding order check:
    // TODO: support non-planar faces and relax this:
    return numVertices >= 4 && numVertices <= 8 &&
        allFacesPlanar() && !hasCoplanarFaces();// && checkWindingOrder();
  }

  // vertices, with index indirection already resolved
  float4 vertices[8];
  // indices, as from the original data
  uint64_t indices[8];
  int numVertices;
};


// ========================================================
// mesh
// ========================================================

struct Mesh
{
  Mesh(const float4 *vertices,
       const uint64_t *indices,
       const dco::UElem *elements,
       size_t numElems)
    : vertices(vertices),
      indices(indices),
      elements(elements),
      numElems(numElems)
  {
    bounds.invalidate();
    for (size_t elemID=0; elemID<numElems; ++elemID) {
      int numVerts = elements[elemID].end-elements[elemID].begin;
      for (int i=0; i<numVerts; ++i) {
        uint64_t idx = indices[elements[elemID].begin+i];
        float4 v = vertices[idx];
        bounds.insert(v.xyz());
      }
    }
  }

  const float4 *vertices;
  const uint64_t *indices;
  const dco::UElem *elements;
  size_t numElems;

  aabb bounds;
};

struct ElemPair {
  // IDs of left/right element
  uint64_t L,R;
};

} // conn

// interface:
std::vector<uint64_t> computeFaceConnectivity(const conn::Mesh &mesh);

std::vector<basic_triangle<3,float>> computeShell(
    const conn::Mesh &mesh, const uint64_t *faceNeighbors);

} // namespace visionaray
