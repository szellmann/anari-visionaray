
#pragma once

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <visionaray/morton.h>
#include "DeviceCopyableObjects.h"

namespace visionaray {

namespace conn {

__device__ static const int Tet[4][3] {
  { 1, 3, 2 },
  { 0, 2, 3 },
  { 0, 3, 1 },
  { 0, 1, 2 },
};

__device__ static const int Pyr[5][4] {
  { 0, 1, 2, 3 },
  { 0, 4, 1, -1 },
  { 2, 4, 3, -1 },
  { 0, 3, 4, -1 },
  { 1, 4, 2, -1 },
};

__device__ static const int Wed[5][4] {
  { 0, 1, 3, 4 },
  { 0, 2, 1, -1 },
  { 3, 4, 5, -1 },
  { 0, 3, 2, 5 },
  { 1, 2, 4, 5 },
};

__device__ static const int Hex[6][4] {
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

  // three indices
  inline UniqueFace makeUniqueFace(uint64_t i1, uint64_t i2, uint64_t i3) const
  {
    uint64_t I[3] = { i1, i2, i3 };
    std::sort(I,I+3);
    UniqueFace f;
    f.i1 = I[0];
    f.i2 = I[1];
    f.i3 = I[2];
    return f;
  }

  // four indices
  inline UniqueFace makeUniqueFace(
      uint64_t i1, uint64_t i2, uint64_t i3, uint64_t i4) const
  {
    uint64_t I[4] = { i1, i2, i3, i4 };
    std::sort(I,I+4);
    UniqueFace f;
    f.i1 = I[0];
    f.i2 = I[1];
    f.i3 = I[2];
    return f;
  }

  // three vertices, generate indices on morton curve
  inline UniqueFace makeUniqueFace(
      const float4 &v1, const float4 &v2, const float4 &v3) const
  {
    uint64_t morton[3] = {
      quantize(v1.xyz()),
      quantize(v2.xyz()),
      quantize(v3.xyz())
    };
    std::sort(morton,morton+3);
    UniqueFace f;
    f.i1 = morton[0];
    f.i2 = morton[1];
    f.i3 = morton[2];
    return f;
  }

  // four vertices, generate indices on morton curve
  inline UniqueFace makeUniqueFace(
      const float4 &v1, const float4 &v2, const float4 &v3, const float4 &v4) const
  {
    uint64_t morton[4] = {
      quantize(v1.xyz()),
      quantize(v2.xyz()),
      quantize(v3.xyz()),
      quantize(v4.xyz())
    };
    std::sort(morton,morton+4);
    UniqueFace f;
    f.i1 = morton[0];
    f.i2 = morton[1];
    f.i3 = morton[2];
    return f;
  }

  inline uint64_t quantize(float3 p) const {
    p = (p - bounds.min) / bounds.size();
    p.x *= 0x400000;
    p.y *= 0x200000;
    p.z *= 0x200000;
    typedef unsigned long long ulonglong;
    return morton_encode3D(ulonglong(p.x), ulonglong(p.y), ulonglong(p.z));
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

struct asTet {
  asTet(const float4 *verts) : vertices(verts)
  {}

  const float4 *vertices;
  //Face
};

template <typename Lambda>
void for_each_uface(const Mesh &mesh, const dco::UElem &elem, Lambda lambda) {
  size_t numVerts = elem.end-elem.begin;
  float4 v[8];
  uint64_t I[8];
  for (int i=0; i<numVerts; ++i) {
    uint64_t idx = elem.indexBuffer[elem.begin+i];
    v[i] = elem.vertexBuffer[idx];
    I[i] = idx;
  }
  //#define ARR v
  #define ARR I
  if (numVerts == 4) { // tet
    lambda(mesh.makeUniqueFace(ARR[Tet[0][0]],ARR[Tet[0][1]],ARR[Tet[0][2]]));
    lambda(mesh.makeUniqueFace(ARR[Tet[1][0]],ARR[Tet[1][1]],ARR[Tet[1][2]]));
    lambda(mesh.makeUniqueFace(ARR[Tet[2][0]],ARR[Tet[2][1]],ARR[Tet[2][2]]));
    lambda(mesh.makeUniqueFace(ARR[Tet[3][0]],ARR[Tet[3][1]],ARR[Tet[3][2]]));
  } else if (numVerts == 5) { // pyr
    lambda(mesh.makeUniqueFace(ARR[Pyr[0][0]],ARR[Pyr[0][1]],ARR[Pyr[0][2]],ARR[Pyr[0][3]]));
    lambda(mesh.makeUniqueFace(ARR[Pyr[1][0]],ARR[Pyr[1][1]],ARR[Pyr[1][2]]));
    lambda(mesh.makeUniqueFace(ARR[Pyr[2][0]],ARR[Pyr[2][1]],ARR[Pyr[2][2]]));
    lambda(mesh.makeUniqueFace(ARR[Pyr[3][0]],ARR[Pyr[3][1]],ARR[Pyr[3][2]]));
    lambda(mesh.makeUniqueFace(ARR[Pyr[4][0]],ARR[Pyr[4][1]],ARR[Pyr[4][2]]));
  } else if (numVerts == 6) { // wedge
    lambda(mesh.makeUniqueFace(ARR[Wed[0][0]],ARR[Wed[0][1]],ARR[Wed[0][2]],ARR[Wed[0][3]]));
    lambda(mesh.makeUniqueFace(ARR[Wed[1][0]],ARR[Wed[1][1]],ARR[Wed[1][2]]));
    lambda(mesh.makeUniqueFace(ARR[Wed[2][0]],ARR[Wed[2][1]],ARR[Wed[2][2]]));
    lambda(mesh.makeUniqueFace(ARR[Wed[3][0]],ARR[Wed[3][1]],ARR[Wed[3][2]],ARR[Wed[3][3]]));
    lambda(mesh.makeUniqueFace(ARR[Wed[4][0]],ARR[Wed[4][1]],ARR[Wed[4][2]],ARR[Wed[4][3]]));
  } else if (numVerts == 8) { // hex
    lambda(mesh.makeUniqueFace(ARR[Hex[0][0]],ARR[Hex[0][1]],ARR[Hex[0][2]],ARR[Hex[0][3]]));
    lambda(mesh.makeUniqueFace(ARR[Hex[1][0]],ARR[Hex[1][1]],ARR[Hex[1][2]],ARR[Hex[1][3]]));
    lambda(mesh.makeUniqueFace(ARR[Hex[2][0]],ARR[Hex[2][1]],ARR[Hex[2][2]],ARR[Hex[2][3]]));
    lambda(mesh.makeUniqueFace(ARR[Hex[3][0]],ARR[Hex[3][1]],ARR[Hex[3][2]],ARR[Hex[3][3]]));
    lambda(mesh.makeUniqueFace(ARR[Hex[4][0]],ARR[Hex[4][1]],ARR[Hex[4][2]],ARR[Hex[4][3]]));
    lambda(mesh.makeUniqueFace(ARR[Hex[5][0]],ARR[Hex[5][1]],ARR[Hex[5][2]],ARR[Hex[5][3]]));
  }
}

} // conn

static
std::vector<uint64_t> computeFaceConnectivity(const conn::Mesh &mesh)
{
  size_t numElems = mesh.numElems;

  std::unordered_map<conn::UniqueFace, conn::ElemPair, conn::UniqueFaceHash, conn::UniqueFaceEqual> face2elems;

  for (size_t elemID=0; elemID<numElems; ++elemID) {
    // get on host (incl. buffers):
    dco::UElem elem(mesh.elements[elemID]);
    elem.vertexBuffer = mesh.vertices;
    elem.indexBuffer = mesh.indices;

    conn::for_each_uface(mesh, mesh.elements[elemID], [&](conn::UniqueFace face) {
      auto it = face2elems.find(face);
      if (it == face2elems.end()) {
        face2elems.insert({face,{elemID,~0ull}});
      } else {
        auto &ep = it->second;
        ep.R = elemID;
      }
    });
  }

  std::vector<uint64_t> faceNeighbors;

  for (size_t elemID=0; elemID<numElems; ++elemID) {
    // get on host (incl. buffers):
    dco::UElem elem(mesh.elements[elemID]);
    elem.vertexBuffer = mesh.vertices;
    elem.indexBuffer = mesh.indices;

    int i=0;

    conn::for_each_uface(mesh, mesh.elements[elemID], [&](conn::UniqueFace face) {
      auto it = face2elems.find(face);
      assert(it != face2elems.end());
      const auto &ep = it->second;
      if (ep.R==~0ull) {
        assert(ep.L==elemID);
        faceNeighbors.push_back(~0ull);
      } else {
        if (ep.L == elemID) {
          faceNeighbors.push_back(ep.R);
        } else {
          assert(ep.R == elemID);
          faceNeighbors.push_back(ep.L);
        }
      }
      i++;
    });

    for (int j=i; j<6; ++j) {
      faceNeighbors.push_back(~0ull);
    }
  }

  return faceNeighbors;
}

static
std::vector<basic_triangle<3,float>> computeShell(const conn::Mesh &mesh,
                                                  const uint64_t *faceNeighbors)
{
  size_t numElems = mesh.numElems;

  std::vector<basic_triangle<3,float>> result;

  for (size_t elemID=0; elemID<numElems; ++elemID) {
    // get on host (incl. buffers):
    dco::UElem elem(mesh.elements[elemID]);
    elem.vertexBuffer = mesh.vertices;
    elem.indexBuffer = mesh.indices;

    size_t numVerts = elem.end-elem.begin;

    float4 v[8];
    for (int i=0; i<numVerts; ++i) {
      uint64_t idx = elem.indexBuffer[elem.begin+i];
      v[i] = elem.vertexBuffer[idx];
    }

    auto nb = faceNeighbors + elemID*6;
    for (int i=0; i<8; ++i) {
      auto addTriangle = [&](const float4 v1, const float4 v2, const float4 v3) {
        basic_triangle<3,float> tri;
        tri.prim_id = (unsigned)result.size();
        tri.geom_id = elemID;
        tri.v1 = v1.xyz();
        tri.e2 = v2.xyz()-v1.xyz();
        tri.e1 = v3.xyz()-v1.xyz();
        result.push_back(tri);
      };

      if (nb[i] == ~0ull) {
        if (numVerts == 4) { // tet
          using conn::Tet;
          if (i == 0) addTriangle(v[Tet[0][0]],v[Tet[0][1]],v[Tet[0][2]]);
          if (i == 1) addTriangle(v[Tet[1][0]],v[Tet[1][1]],v[Tet[1][2]]);
          if (i == 2) addTriangle(v[Tet[2][0]],v[Tet[2][1]],v[Tet[2][2]]);
          if (i == 3) addTriangle(v[Tet[3][0]],v[Tet[3][1]],v[Tet[3][2]]);
        } else if (numVerts == 5) { // pyr
          using conn::Pyr;
          if (i == 0) {
            addTriangle(v[Pyr[0][0]],v[Pyr[0][1]],v[Pyr[0][2]]);
            addTriangle(v[Pyr[0][0]],v[Pyr[0][2]],v[Pyr[0][3]]);
          }
          if (i == 1) {
            addTriangle(v[Pyr[1][0]],v[Pyr[1][1]],v[Pyr[1][2]]);
          }
          if (i == 2) {
            addTriangle(v[Pyr[2][0]],v[Pyr[2][1]],v[Pyr[2][2]]);
          }
          if (i == 3) {
            addTriangle(v[Pyr[3][0]],v[Pyr[3][1]],v[Pyr[3][2]]);
          }
          if (i == 4) {
            addTriangle(v[Pyr[4][0]],v[Pyr[4][1]],v[Pyr[4][2]]);
          }
        } else if (numVerts == 6) { // wedge
          using conn::Wed;
          if (i == 0) {
            addTriangle(v[Wed[0][0]],v[Wed[0][1]],v[Wed[0][2]]);
            addTriangle(v[Wed[0][1]],v[Wed[0][3]],v[Wed[0][2]]);
          }
          if (i == 1) {
            addTriangle(v[Wed[1][0]],v[Wed[1][1]],v[Wed[1][2]]);
          }
          if (i == 2) {
            addTriangle(v[Wed[2][0]],v[Wed[2][1]],v[Wed[2][2]]);
          }
          if (i == 3) {
            addTriangle(v[Wed[3][0]],v[Wed[3][1]],v[Wed[3][2]]);
            addTriangle(v[Wed[3][1]],v[Wed[3][3]],v[Wed[3][2]]);
          }
          if (i == 4) {
            addTriangle(v[Wed[4][0]],v[Wed[4][1]],v[Wed[4][2]]);
            addTriangle(v[Wed[4][1]],v[Wed[4][3]],v[Wed[4][2]]);
          }
        } else if (numVerts == 8) { // hex
          using conn::Hex;
          if (i == 0) {
            addTriangle(v[Hex[0][0]],v[Hex[0][1]],v[Hex[0][2]]);
            addTriangle(v[Hex[0][1]],v[Hex[0][3]],v[Hex[0][2]]);
          }
          if (i == 1) {
            addTriangle(v[Hex[1][0]],v[Hex[1][1]],v[Hex[1][2]]);
            addTriangle(v[Hex[1][1]],v[Hex[1][3]],v[Hex[1][2]]);
          }
          if (i == 2) {
            addTriangle(v[Hex[2][0]],v[Hex[2][1]],v[Hex[2][2]]);
            addTriangle(v[Hex[2][1]],v[Hex[2][3]],v[Hex[2][2]]);
          }
          if (i == 3) {
            addTriangle(v[Hex[3][0]],v[Hex[3][1]],v[Hex[3][2]]);
            addTriangle(v[Hex[3][1]],v[Hex[3][3]],v[Hex[3][2]]);
          }
          if (i == 4) {
            addTriangle(v[Hex[4][0]],v[Hex[4][1]],v[Hex[4][2]]);
            addTriangle(v[Hex[4][1]],v[Hex[4][3]],v[Hex[4][2]]);
          }
          if (i == 5) {
            addTriangle(v[Hex[5][0]],v[Hex[5][1]],v[Hex[5][2]]);
            addTriangle(v[Hex[5][1]],v[Hex[5][3]],v[Hex[5][2]]);
          }
        }
      }
    }
  }

  return result;
}

} // visionaray
