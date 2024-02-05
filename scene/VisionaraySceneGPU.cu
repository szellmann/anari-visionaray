
#include "VisionarayScene.h"
#include "VisionaraySceneGPU.h"

namespace visionaray {

struct VisionaraySceneGPU::Impl
{
  typedef cuda_index_bvh<basic_triangle<3,float>> TriangleBVH;
  typedef cuda_index_bvh<basic_triangle<3,float>> QuadBVH;
  typedef cuda_index_bvh<basic_sphere<float>>     SphereBVH;
  typedef cuda_index_bvh<basic_cylinder<float>>   CylinderBVH;
  typedef cuda_index_bvh<dco::ISOSurface>         ISOSurfaceBVH;
  typedef cuda_index_bvh<dco::Volume>             VolumeBVH;

  typedef cuda_index_bvh<dco::BLS> TLS;
  typedef cuda_index_bvh<dco::WorldBLS> WorldTLS;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH>   triangleBLSs;
    aligned_vector<QuadBVH>       quadBLSs;
    aligned_vector<SphereBVH>     sphereBLSs;
    aligned_vector<CylinderBVH>   cylinderBLSs;
    aligned_vector<ISOSurfaceBVH> isoSurfaceBLSs;
    aligned_vector<VolumeBVH>     volumeBLSs;
  } m_accelStorage;

  TLS m_TLS;
  WorldTLS m_worldTLS;

  VisionaraySceneImpl *parent{nullptr};

  bool isGroup() const {
    return parent->type == VisionaraySceneImpl::Group;
  }

  bool isWorld() const {
    return parent->type == VisionaraySceneImpl::World;
  }
};

VisionaraySceneGPU::VisionaraySceneGPU(VisionaraySceneImpl *cpuScene)
  : m_impl(new Impl)
{
  m_impl->parent = cpuScene;
}

VisionaraySceneGPU::~VisionaraySceneGPU()
{
}

bool VisionaraySceneGPU::isValid() const
{
  if (m_impl->isWorld())
    return m_impl->m_worldTLS.num_nodes() > 0;
  else
    return m_impl->m_TLS.num_nodes() > 0;
}

aabb VisionaraySceneGPU::getBounds() const
{
  aabb bounds;
  bounds.invalidate();
  if (m_impl->isGroup() && m_impl->m_TLS.num_nodes() > 0) {
    bvh_node rootNode;
    CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                              thrust::raw_pointer_cast(m_impl->m_TLS.nodes().data()),
                              sizeof(rootNode),
                              cudaMemcpyDeviceToHost));
    bounds = rootNode.get_bounds();
  } else if (m_impl->isWorld() && m_impl->m_worldTLS.num_nodes() > 0) {
    bvh_node rootNode;
    CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                              thrust::raw_pointer_cast(m_impl->m_worldTLS.nodes().data()),
                              sizeof(rootNode),
                              cudaMemcpyDeviceToHost));
    bounds = rootNode.get_bounds();
  }
  return bounds;
}

void VisionaraySceneGPU::commit()
{
  using CPU = VisionaraySceneImpl;
  using GPU = Impl;

  // create refs (on device!):
  unsigned triangleCount = 0;
  unsigned quadCount = 0;
  unsigned sphereCount = 0;
  unsigned cylinderCount = 0;
  unsigned isoCount = 0;
  unsigned volumeCount = 0;
  unsigned instanceCount = 0;

  for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    switch (geom.type) {
      case dco::Geometry::Triangle:
        triangleCount++;
        break;
      case dco::Geometry::Quad:
        quadCount++;
        break;
      case dco::Geometry::Sphere:
        sphereCount++;
        break;
      case dco::Geometry::Cylinder:
        cylinderCount++;
        break;
      case dco::Geometry::ISOSurface:
        isoCount++;
        break;
      case dco::Geometry::Volume:
        volumeCount++;
        break;
      case dco::Geometry::Instance:
      default:
        break;
    }
  }

  m_impl->m_accelStorage.triangleBLSs.resize(triangleCount);
  m_impl->m_accelStorage.quadBLSs.resize(quadCount);
  m_impl->m_accelStorage.sphereBLSs.resize(sphereCount);
  m_impl->m_accelStorage.cylinderBLSs.resize(cylinderCount);
  m_impl->m_accelStorage.isoSurfaceBLSs.resize(isoCount);
  m_impl->m_accelStorage.volumeBLSs.resize(volumeCount);
  // No instance storage: instance BLSs are the TLSs of child scenes

  // first, build BLSs
  triangleCount = quadCount = sphereCount = cylinderCount = isoCount = volumeCount = 0;
  for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    binned_sah_builder builder;

    if (geom.type == dco::Geometry::Triangle) {
      unsigned index = triangleCount++;
      builder.enable_spatial_splits(true);
      // Until we have a high-quality GPU builder, do that on the CPU!
      std::vector<basic_triangle<3,float>> hostData(geom.asTriangle.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asTriangle.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      auto accelStorage = builder.build(
        CPU::TriangleBVH{}, hostData.data(), hostData.size());
      m_impl->m_accelStorage.triangleBLSs[index]
        = GPU::TriangleBVH(accelStorage);
    } else if (geom.type == dco::Geometry::Quad) {
      unsigned index = quadCount++;
      builder.enable_spatial_splits(true);
      // Until we have a high-quality GPU builder, do that on the CPU!
      std::vector<basic_triangle<3,float>> hostData(geom.asQuad.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asQuad.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      auto accelStorage = builder.build(
        CPU::TriangleBVH{}, hostData.data(), hostData.size());
      m_impl->m_accelStorage.quadBLSs[index]
        = GPU::TriangleBVH(accelStorage);
    } else if (geom.type == dco::Geometry::Sphere) {
      unsigned index = sphereCount++;
      builder.enable_spatial_splits(true);
      // Until we have a high-quality GPU builder, do that on the CPU!
      std::vector<basic_sphere<float>> hostData(geom.asSphere.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asSphere.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      auto accelStorage = builder.build(
        CPU::SphereBVH{}, hostData.data(), hostData.size());
      m_impl->m_accelStorage.sphereBLSs[index]
        = GPU::SphereBVH(accelStorage);
    } else if (geom.type == dco::Geometry::Cylinder) {
      unsigned index = cylinderCount++;
      builder.enable_spatial_splits(false); // no spatial splits for cyls yet!
      // Until we have a high-quality GPU builder, do that on the CPU!
      std::vector<basic_cylinder<float>> hostData(geom.asCylinder.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asCylinder.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      auto accelStorage = builder.build(
        CPU::CylinderBVH{}, hostData.data(), hostData.size());
      m_impl->m_accelStorage.cylinderBLSs[index]
        = GPU::CylinderBVH(accelStorage);
    } else if (geom.type == dco::Geometry::ISOSurface) {
      unsigned index = isoCount++;
      builder.enable_spatial_splits(false); // no spatial splits for ISOs
      // Until we have a high-quality GPU builder, do that on the CPU!
      auto accelStorage = builder.build(
        CPU::ISOSurfaceBVH{}, &geom.asISOSurface.data, 1);
      m_impl->m_accelStorage.isoSurfaceBLSs[index]
        = GPU::ISOSurfaceBVH(accelStorage);
    } else if (geom.type == dco::Geometry::Volume) {
      unsigned index = volumeCount++;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs
      auto accelStorage = builder.build(CPU::VolumeBVH{}, &geom.asVolume.data, 1);
      m_impl->m_accelStorage.volumeBLSs[index]
        = GPU::VolumeBVH(accelStorage);
    } else if (geom.type == dco::Geometry::Instance) {
      instanceCount++;
    }
  }

  m_impl->parent->m_BLSs.clear();
  m_impl->parent->m_worldBLSs.clear();

  // now initialize BVH refs for use in shader code:
  triangleCount = quadCount = sphereCount = cylinderCount = isoCount = volumeCount = 0;
  for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
    if (!dco::validHandle(geomID)) continue;

    const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
    if (!geom.isValid()) continue;

    if (m_impl->parent->type == CPU::World) {
      dco::WorldBLS bls;
      bls.blsID = m_impl->parent->m_worldBLSs.alloc(bls);

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        bls.type = dco::BLS::Triangle;
        bls.asTriangle = m_impl->m_accelStorage.triangleBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        bls.type = dco::BLS::Quad;
        bls.asQuad = m_impl->m_accelStorage.quadBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        bls.type = dco::BLS::Sphere;
        bls.asSphere = m_impl->m_accelStorage.sphereBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
        bls.asCylinder = m_impl->m_accelStorage.cylinderBLSs[index].ref();
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
        bls.asISOSurface = m_impl->m_accelStorage.isoSurfaceBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Volume) {
        unsigned index = volumeCount++;
        bls.type = dco::BLS::Volume;
        bls.asVolume = m_impl->m_accelStorage.volumeBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Instance) {
        instanceCount++;
        bls.type = dco::BLS::Instance;
        bls.asInstance = geom.asInstance.data.instBVH;
        bls.asInstance.set_inst_id(geom.asInstance.data.instID);
      }
      m_impl->parent->m_worldBLSs.update(bls.blsID, bls);
    } else {
      dco::BLS bls;
      bls.blsID = m_impl->parent->m_BLSs.alloc(bls);

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        bls.type = dco::BLS::Triangle;
        bls.asTriangle = m_impl->m_accelStorage.triangleBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        bls.type = dco::BLS::Quad;
        bls.asQuad = m_impl->m_accelStorage.quadBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        bls.type = dco::BLS::Sphere;
        bls.asSphere = m_impl->m_accelStorage.sphereBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
        bls.asCylinder = m_impl->m_accelStorage.cylinderBLSs[index].ref();
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
        bls.asISOSurface = m_impl->m_accelStorage.isoSurfaceBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Volume) {
        unsigned index = volumeCount++;
        bls.type = dco::BLS::Volume;
        bls.asVolume = m_impl->m_accelStorage.volumeBLSs[index].ref();
      }
      m_impl->parent->m_BLSs.update(bls.blsID, bls);
    }
  }

  // Build TLS
  lbvh_builder tlsBuilder;
  if (m_impl->parent->type == CPU::World) {
    m_impl->m_worldTLS = tlsBuilder.build(
        GPU::WorldTLS{}, m_impl->parent->m_worldBLSs.devicePtr(),
        m_impl->parent->m_worldBLSs.size());
  } else {
    m_impl->m_TLS = tlsBuilder.build(
        GPU::TLS{}, m_impl->parent->m_BLSs.devicePtr(),
        m_impl->parent->m_BLSs.size());
  }

  // World: build flat list of lights
  if (m_impl->parent->type == CPU::World) {
    m_impl->parent->m_allLights.clear();

    // world lights
    for (unsigned i=0; i<m_impl->parent->m_lights.size(); ++i)
      m_impl->parent->m_allLights.push_back(m_impl->parent->m_lights[i]);

    // instanced lights
    for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
      if (!geom.isValid()) continue;

      if (geom.type != dco::Geometry::Instance) continue;

      dco::Instance inst = geom.asInstance.data;
      dco::Group group = deviceState()->dcos.groups[inst.groupID];

      std::vector<dco::Handle> groupLights(group.numLights);
      CUDA_SAFE_CALL(cudaMemcpy(groupLights.data(), group.lights,
                                group.numLights*sizeof(dco::Handle),
                                cudaMemcpyDefault));
      for (unsigned i=0; i<group.numLights; ++i)
        m_impl->parent->m_allLights.push_back(groupLights[i]);
    }
  }
}

void VisionaraySceneGPU::dispatch()
{
  if (m_impl->parent->type == VisionaraySceneImpl::World) {
    deviceState()->dcos.TLSs.update(
        m_impl->parent->m_worldID, m_impl->m_worldTLS.ref());
    deviceState()->dcos.worldEPS.update(m_impl->parent->m_worldID,
        m_impl->parent->getWorldEPS());

    dco::World world; // TODO: move TLS and EPS in here!
    world.numLights = m_impl->parent->m_allLights.size();
    world.allLights = m_impl->parent->m_allLights.devicePtr();
    deviceState()->dcos.worlds.update(m_impl->parent->m_worldID, world);
  }

  // Dispatch group
  if (m_impl->parent->type == VisionaraySceneImpl::Group) {
    dco::Group group;
    group.groupID = m_impl->parent->m_groupID;
    group.numBLSs = m_impl->parent->m_BLSs.size();
    group.BLSs = m_impl->parent->m_BLSs.devicePtr();
    group.numGeoms = m_impl->parent->m_geometries.size();
    group.geoms = m_impl->parent->m_geometries.devicePtr();
    group.numMaterials = m_impl->parent->m_materials.size();
    group.materials = m_impl->parent->m_materials.devicePtr();
    group.numLights = m_impl->parent->m_lights.size();
    group.lights = m_impl->parent->m_lights.devicePtr();
    deviceState()->dcos.groups.update(m_impl->parent->m_groupID, group);
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.TLSs = deviceState()->dcos.TLSs.devicePtr();
  deviceState()->onDevice.worldEPS = deviceState()->dcos.worldEPS.devicePtr();
  deviceState()->onDevice.groups = deviceState()->dcos.groups.devicePtr();
  if (m_impl->parent->type == VisionaraySceneImpl::World) {
    deviceState()->onDevice.worlds = deviceState()->dcos.worlds.devicePtr();
  }
}

void VisionaraySceneGPU::attachGeometry(dco::Geometry geom, unsigned geomID)
{
  m_impl->parent->m_geometries.set(geomID, geom.geomID);

  // Patch geomID into scene primitives
  // (first copy to CPU, patch there, then copy back...)
  if (geom.type == dco::Geometry::Triangle) {
    if (geom.asTriangle.len > 0) {
      std::vector<basic_triangle<3,float>> hostData(geom.asTriangle.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asTriangle.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.asTriangle.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy(geom.asTriangle.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    }
  } else if (geom.type == dco::Geometry::Quad) {
    if (geom.asQuad.len > 0) {
      std::vector<basic_triangle<3,float>> hostData(geom.asQuad.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asQuad.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.asQuad.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy(geom.asQuad.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    }
  } else if (geom.type == dco::Geometry::Sphere) {
    if (geom.asSphere.len > 0) {
      std::vector<basic_sphere<float>> hostData(geom.asSphere.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asSphere.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.asSphere.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy(geom.asSphere.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    }
  } else if (geom.type == dco::Geometry::Cylinder) {
    if (geom.asCylinder.len > 0) {
      std::vector<basic_cylinder<float>> hostData(geom.asCylinder.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.asCylinder.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.asCylinder.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy(geom.asCylinder.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    }
  } else if (geom.type == dco::Geometry::ISOSurface) {
    geom.asISOSurface.data.isoID = geomID;
    geom.asISOSurface.data.geomID = geomID;
  } else if (geom.type == dco::Geometry::Volume) {
    geom.asVolume.data.geomID = geomID;
  }


  m_impl->parent->m_geometries.set(geomID, geom.geomID);
  deviceState()->dcos.geometries.update(geom.geomID, geom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

cuda_index_bvh<dco::BLS>::bvh_inst VisionaraySceneGPU::instBVH(mat4x3 xfm)
{
  return m_impl->m_TLS.inst(xfm);
}

VisionarayGlobalState *VisionaraySceneGPU::deviceState()
{
  return m_impl->parent->deviceState();
}

} // namespace visionaray
