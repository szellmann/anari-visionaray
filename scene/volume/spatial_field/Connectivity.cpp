
#include "Connectivity.h"

namespace visionaray {

std::vector<uint64_t> computeFaceConnectivity(const conn::Mesh &mesh)
{
  size_t numElems = mesh.numElems;

  std::unordered_map<conn::UniqueFace, conn::ElemPair, conn::UniqueFaceHash, conn::UniqueFaceEqual> face2elems;

  for (size_t elemID=0; elemID<numElems; ++elemID) {
    // get on host (incl. buffers):
    dco::UElem elem(mesh.elements[elemID]);
    elem.vertexBuffer = mesh.vertices;
    elem.indexBuffer = mesh.indices;

    conn::UElem cElem(elem);
    if (!cElem.isValid()) continue;

    for (int i=0; i<cElem.numFaces(); ++i) {
      conn::UniqueFace face = cElem.uniqueFace(i);
      auto it = face2elems.find(face);
      if (it == face2elems.end()) {
        face2elems.insert({face,{elemID,~0ull}});
      } else {
        auto &ep = it->second;
        ep.R = elemID;
      }
    }
  }

  std::vector<uint64_t> faceNeighbors;

  for (size_t elemID=0; elemID<numElems; ++elemID) {
    // get on host (incl. buffers):
    dco::UElem elem(mesh.elements[elemID]);
    elem.vertexBuffer = mesh.vertices;
    elem.indexBuffer = mesh.indices;

    conn::UElem cElem(elem);
    if (!cElem.isValid()) {
      for (int i=0; i<6; ++i) {
        faceNeighbors.push_back(~0ull);
      }
      continue;
    }

    for (int i=0; i<cElem.numFaces(); ++i) {
      conn::UniqueFace face = cElem.uniqueFace(i);
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
    }

    for (int i=cElem.numFaces(); i<6; ++i) {
      faceNeighbors.push_back(~0ull);
    }
  }

  return faceNeighbors;
}

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

    auto nb = faceNeighbors + elemID*6;

    conn::UElem cElem(elem);

    if (!cElem.isValid()) continue;

    for (int i=0; i<cElem.numFaces(); ++i) {
      if (nb[i] == ~0ull) {
        const auto &face = cElem.face(i);
        for (int j=0; j<face.numTriangles(); ++j) {
          auto tri = face.triangle(j);
          tri.prim_id = (unsigned)result.size();
          tri.geom_id = elemID;
          result.push_back(tri);
        }
      }
    }
  }

  return result;
}

} // namespace visionaray
