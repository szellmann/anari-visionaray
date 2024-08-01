
#include "VisionarayScene.h"
#include "VisionaraySceneGPU.h"

namespace visionaray {

struct VisionaraySceneGPU::Impl
{
  typedef cuda_index_bvh<basic_triangle<3,float>> TriangleBVH;
  typedef cuda_index_bvh<basic_triangle<3,float>> QuadBVH;
  typedef cuda_index_bvh<basic_sphere<float>>     SphereBVH;
  typedef cuda_index_bvh<dco::Cone>               ConeBVH;
  typedef cuda_index_bvh<basic_cylinder<float>>   CylinderBVH;
  typedef cuda_index_bvh<dco::BezierCurve>        BezierCurveBVH;
  typedef cuda_index_bvh<dco::ISOSurface>         ISOSurfaceBVH;
  typedef cuda_index_bvh<dco::Volume>             VolumeBVH;

  typedef cuda_index_bvh<dco::BLS> TLS;
  typedef cuda_index_bvh<dco::Instance> WorldTLS;

  // Accel storage //
  struct {
    aligned_vector<TriangleBVH>    triangleBLSs;
    aligned_vector<QuadBVH>        quadBLSs;
    aligned_vector<SphereBVH>      sphereBLSs;
    aligned_vector<ConeBVH>        coneBLSs;
    aligned_vector<CylinderBVH>    cylinderBLSs;
    aligned_vector<BezierCurveBVH> bezierCurveBLSs;
    aligned_vector<ISOSurfaceBVH>  isoSurfaceBLSs;
    aligned_vector<VolumeBVH>      volumeBLSs;
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
                              m_impl->m_TLS.nodes().data(),
                              sizeof(rootNode),
                              cudaMemcpyDeviceToHost));
    bounds = rootNode.get_bounds();
  } else if (m_impl->isWorld() && m_impl->m_worldTLS.num_nodes() > 0) {
    bvh_node rootNode;
    CUDA_SAFE_CALL(cudaMemcpy(&rootNode,
                              m_impl->m_worldTLS.nodes().data(),
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

  if (m_impl->parent->type == CPU::World) {
    m_impl->parent->m_worldBLSs.clear();

    for (const dco::Handle &instID : m_impl->parent->m_instances) {
      if (!dco::validHandle(instID)) continue;

      dco::Instance inst;
      CUDA_SAFE_CALL(cudaMemcpy(&inst, deviceState()->onDevice.instances+instID,
                                sizeof(inst), cudaMemcpyDefault));

      m_impl->parent->m_worldBLSs.alloc(inst);
    }

    // Build TLS
    lbvh_builder tlsBuilder;
    m_impl->m_worldTLS = tlsBuilder.build(
        GPU::WorldTLS{}, m_impl->parent->m_worldBLSs.devicePtr(),
        m_impl->parent->m_worldBLSs.size());

    // Build flat list of lights
    m_impl->parent->m_allLights.clear();

    // world lights
    for (unsigned i=0; i<m_impl->parent->m_lights.size(); ++i)
      m_impl->parent->m_allLights.push_back(m_impl->parent->m_lights[i]);

    // instanced lights
    for (const dco::Handle &instID : m_impl->parent->m_instances) {
      if (!dco::validHandle(instID)) continue;

      dco::Instance inst;
      CUDA_SAFE_CALL(cudaMemcpy(&inst, deviceState()->onDevice.instances+instID,
                                sizeof(inst), cudaMemcpyDefault));

      if (!dco::validHandle(inst.groupID)) continue;
      dco::Group group = deviceState()->dcos.groups[inst.groupID];

      std::vector<dco::Handle> groupLights(group.numLights);
      CUDA_SAFE_CALL(cudaMemcpy(groupLights.data(), group.lights,
                                group.numLights*sizeof(dco::Handle),
                                cudaMemcpyDefault));
      for (unsigned i=0; i<group.numLights; ++i)
        m_impl->parent->m_allLights.push_back(groupLights[i]);
    }
  } else {
    // create refs (on device!):
    unsigned triangleCount = 0;
    unsigned quadCount = 0;
    unsigned sphereCount = 0;
    unsigned coneCount = 0;
    unsigned cylinderCount = 0;
    unsigned bezierCurveCount = 0;
    unsigned isoCount = 0;
    unsigned volumeCount = 0;

    for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      dco::Geometry geom = deviceState()->dcos.geometries[geomID];
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
        case dco::Geometry::Cone:
          coneCount++;
          break;
        case dco::Geometry::Cylinder:
          cylinderCount++;
          break;
        case dco::Geometry::BezierCurve:
          bezierCurveCount++;
          break;
        case dco::Geometry::ISOSurface:
          isoCount++;
          break;
        default:
          break;
      }
    }

    for (const dco::Handle &volID : m_impl->parent->m_volumes) {
      if (!dco::validHandle(volID)) continue;
      volumeCount++;
    }

    m_impl->m_accelStorage.triangleBLSs.resize(triangleCount);
    m_impl->m_accelStorage.quadBLSs.resize(quadCount);
    m_impl->m_accelStorage.sphereBLSs.resize(sphereCount);
    m_impl->m_accelStorage.coneBLSs.resize(coneCount);
    m_impl->m_accelStorage.cylinderBLSs.resize(cylinderCount);
    m_impl->m_accelStorage.bezierCurveBLSs.resize(bezierCurveCount);
    m_impl->m_accelStorage.isoSurfaceBLSs.resize(isoCount);
    m_impl->m_accelStorage.volumeBLSs.resize(volumeCount);

    // first, build BLSs
    triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                  = bezierCurveCount = isoCount = volumeCount = 0;
    for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
      if (!geom.isValid()) continue;

      binned_sah_builder builder;

      if (geom.type == dco::Geometry::Triangle) {
        unsigned index = triangleCount++;
        builder.enable_spatial_splits(true);
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<basic_triangle<3,float>> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::TriangleBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.triangleBLSs[index]
          = GPU::TriangleBVH(accelStorage);
      } else if (geom.type == dco::Geometry::Quad) {
        unsigned index = quadCount++;
        builder.enable_spatial_splits(true);
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<basic_triangle<3,float>> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::TriangleBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.quadBLSs[index]
          = GPU::TriangleBVH(accelStorage);
      } else if (geom.type == dco::Geometry::Sphere) {
        unsigned index = sphereCount++;
        builder.enable_spatial_splits(true);
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<basic_sphere<float>> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::SphereBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.sphereBLSs[index]
          = GPU::SphereBVH(accelStorage);
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        builder.enable_spatial_splits(false); // no spatial splits for cones yet!
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<dco::Cone> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::ConeBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.coneBLSs[index]
          = GPU::ConeBVH(accelStorage);
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        builder.enable_spatial_splits(false); // no spatial splits for cyls yet!
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<basic_cylinder<float>> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::CylinderBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.cylinderBLSs[index]
          = GPU::CylinderBVH(accelStorage);
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        builder.enable_spatial_splits(false); // no spatial splits for bez. curves yet!
        // Until we have a high-quality GPU builder, do that on the CPU!
        std::vector<dco::BezierCurve> hostData(geom.primitives.len);
        CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
            hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::BezierCurveBVH{}, hostData.data(), hostData.size());
        m_impl->m_accelStorage.bezierCurveBLSs[index]
          = GPU::BezierCurveBVH(accelStorage);
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        builder.enable_spatial_splits(false); // no spatial splits for ISOs
        // Until we have a high-quality GPU builder, do that on the CPU!
        dco::ISOSurface iso;
        CUDA_SAFE_CALL(cudaMemcpy(&iso, geom.primitives.data,
            sizeof(iso), cudaMemcpyDeviceToHost));
        auto accelStorage = builder.build(
          CPU::ISOSurfaceBVH{}, &iso, 1);
        m_impl->m_accelStorage.isoSurfaceBLSs[index]
          = GPU::ISOSurfaceBVH(accelStorage);
      }
    }

    for (const dco::Handle &volID : m_impl->parent->m_volumes) {
      if (!dco::validHandle(volID)) continue;

      const dco::Volume &vol = deviceState()->dcos.volumes[volID];

      binned_sah_builder builder;
      unsigned index = volumeCount++;
      builder.enable_spatial_splits(false); // no spatial splits for volumes/aabbs
      auto accelStorage = builder.build(CPU::VolumeBVH{}, &vol, 1);
      m_impl->m_accelStorage.volumeBLSs[index] = GPU::VolumeBVH(accelStorage);
    }

    m_impl->parent->m_BLSs.clear();

    // now initialize BVH refs for use in shader code:
    triangleCount = quadCount = sphereCount = coneCount = cylinderCount
                  = bezierCurveCount = isoCount = volumeCount = 0;
    for (const dco::Handle &geomID : m_impl->parent->m_geometries) {
      if (!dco::validHandle(geomID)) continue;

      const dco::Geometry &geom = deviceState()->dcos.geometries[geomID];
      if (!geom.isValid()) continue;

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
      } else if (geom.type == dco::Geometry::Cone) {
        unsigned index = coneCount++;
        bls.type = dco::BLS::Cone;
        bls.asCone = m_impl->m_accelStorage.coneBLSs[index].ref();
      } else if (geom.type == dco::Geometry::Cylinder) {
        unsigned index = cylinderCount++;
        bls.type = dco::BLS::Cylinder;
        bls.asCylinder = m_impl->m_accelStorage.cylinderBLSs[index].ref();
      } else if (geom.type == dco::Geometry::BezierCurve) {
        unsigned index = bezierCurveCount++;
        bls.type = dco::BLS::BezierCurve;
        bls.asBezierCurve = m_impl->m_accelStorage.bezierCurveBLSs[index].ref();
      } else if (geom.type == dco::Geometry::ISOSurface) {
        unsigned index = isoCount++;
        bls.type = dco::BLS::ISOSurface;
        bls.asISOSurface = m_impl->m_accelStorage.isoSurfaceBLSs[index].ref();
      }
      m_impl->parent->m_BLSs.update(bls.blsID, bls);
    }

    for (const dco::Handle &volID : m_impl->parent->m_volumes) {
      if (!dco::validHandle(volID)) continue;

      dco::BLS bls;
      bls.blsID = m_impl->parent->m_BLSs.alloc(bls);

      unsigned index = volumeCount++;
      bls.type = dco::BLS::Volume;
      bls.asVolume = m_impl->m_accelStorage.volumeBLSs[index].ref();

      m_impl->parent->m_BLSs.update(bls.blsID, bls);
    }

    // Build TLS
    lbvh_builder tlsBuilder;
    m_impl->m_TLS = tlsBuilder.build(
        GPU::TLS{}, m_impl->parent->m_BLSs.devicePtr(),
        m_impl->parent->m_BLSs.size());
  }
}

void VisionaraySceneGPU::dispatch()
{
  if (m_impl->parent->type == VisionaraySceneImpl::World) {
    deviceState()->dcos.TLSs.update(
        m_impl->parent->m_worldID, m_impl->m_worldTLS.ref());

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
    group.objIds = m_impl->parent->m_objIds.devicePtr();
    group.numObjIds = m_impl->parent->m_objIds.size();
    deviceState()->dcos.groups.update(m_impl->parent->m_groupID, group);
  }

  // Upload/set accessible pointers
  deviceState()->onDevice.TLSs = deviceState()->dcos.TLSs.devicePtr();
  deviceState()->onDevice.groups = deviceState()->dcos.groups.devicePtr();
  if (m_impl->parent->type == VisionaraySceneImpl::World) {
    deviceState()->onDevice.worlds = deviceState()->dcos.worlds.devicePtr();
  }
}

void VisionaraySceneGPU::attachGeometry(
    dco::Geometry geom, unsigned geomID, unsigned userID)
{
  m_impl->parent->m_geometries.set(geomID, geom.geomID);
  m_impl->parent->m_objIds.set(geomID, userID);

  // Patch geomID into scene primitives
  // (first copy to CPU, patch there, then copy back...)
  if (geom.primitives.len > 0) {
    if (geom.type == dco::Geometry::Triangle) {
      std::vector<basic_triangle<3,float>> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::Quad) {
      std::vector<basic_triangle<3,float>> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::Sphere) {
      std::vector<basic_sphere<float>> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::Cone) {
      std::vector<dco::Cone> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::Cylinder) {
      std::vector<basic_cylinder<float>> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::BezierCurve) {
      std::vector<dco::BezierCurve> hostData(geom.primitives.len);
      CUDA_SAFE_CALL(cudaMemcpy(hostData.data(), geom.primitives.data,
          hostData.size() * sizeof(hostData[0]), cudaMemcpyDeviceToHost));
      for (size_t i=0;i<geom.primitives.len;++i) {
        hostData[i].geom_id = geomID;
      }
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, hostData.data(),
          hostData.size() * sizeof(hostData[0]), cudaMemcpyHostToDevice));
    } else if (geom.type == dco::Geometry::ISOSurface) {
      dco::ISOSurface iso;
      CUDA_SAFE_CALL(cudaMemcpy(&iso, geom.primitives.data,
                                sizeof(iso), cudaMemcpyDeviceToHost));
      iso.isoID = geomID;
      iso.geomID = geomID;
      CUDA_SAFE_CALL(cudaMemcpy((void *)geom.primitives.data, &iso,
                                sizeof(iso), cudaMemcpyHostToDevice));
    }
  }


  m_impl->parent->m_geometries.set(geomID, geom.geomID);
  deviceState()->dcos.geometries.update(geom.geomID, geom);

  // Upload/set accessible pointers
  deviceState()->onDevice.geometries = deviceState()->dcos.geometries.devicePtr();
}

cuda_index_bvh<dco::BLS>::bvh_ref VisionaraySceneGPU::refBVH()
{
  return m_impl->m_TLS.ref();
}

VisionarayGlobalState *VisionaraySceneGPU::deviceState()
{
  return m_impl->parent->deviceState();
}

} // namespace visionaray
