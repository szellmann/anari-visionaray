// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstdlib>
// visionaray
#if defined(WITH_CUDA)
#include "visionaray/cuda/safe_call.h"
#elif defined(WITH_HIP)
#include "visionaray/hip/safe_call.h"
#endif
#include "visionaray/bvh.h"
// ours
#include "TimeStamp.h"
#include "VisionarayGlobalState.h"

#define BVH_FLAG_ENABLE_SPATIAL_SPLITS 1
#define BVH_FLAG_PREFER_FAST_BUILD     2

namespace visionaray {

template<typename P>
struct DeviceBVH
{
 public:
#if defined(WITH_CUDA)
  typedef cuda_bvh<P> DeviceBVH2;
  typedef cuda_index_bvh<P> DeviceIndexBVH2;
  typedef bvh4<P> DeviceBVH4;
#elif defined(WITH_HIP)
  typedef hip_bvh<P> DeviceBVH2;
  typedef hip_index_bvh<P> DeviceIndexBVH2;
  typedef bvh4<P> DeviceBVH4;
#else
  typedef bvh<P> DeviceBVH2;
  typedef index_bvh<P> DeviceIndexBVH2;
  typedef bvh4<P> DeviceBVH4;
#endif

  DeviceBVH(VisionarayGlobalState *s);

  void update(const P *prims, unsigned numPrims, unsigned flags);

  typename DeviceBVH2::bvh_ref deviceBVH2();
  typename DeviceIndexBVH2::bvh_ref deviceIndexBVH2();
  typename DeviceBVH4::bvh_ref deviceBVH4();

  TimeStamp lastUpdateTime() const;
  TimeStamp lastRebuildTime() const;

 private:
  void rebuildHostBVH2();
  void rebuildHostIndexBVH2();
  void rebuildHostBVH4();

  void rebuildDeviceBVH2();
  void rebuildDeviceIndexBVH2();
  void rebuildDeviceBVH4();

  VisionarayGlobalState *deviceState();

  bvh<P>          m_hostBVH2;
  index_bvh<P>    m_hostIndexBVH2;
  bvh4<P>         m_hostBVH4;

  DeviceBVH2      m_deviceBVH2;
  DeviceIndexBVH2 m_deviceIndexBVH2;
  DeviceBVH4      m_deviceBVH4;

  const P        *m_primitives{nullptr};
  unsigned        m_numPrimitives{0};
  unsigned        m_flags{0};

  VisionarayGlobalState *m_state{nullptr};

  TimeStamp m_lastUpdate{0};
  struct {
    TimeStamp BVH2{0};
    TimeStamp IndexBVH2{0};
    TimeStamp BVH4{0};
  } m_hostRebuild, m_deviceRebuild;
};


// ========================================================
// Impl
// ========================================================

template<typename P>
DeviceBVH<P>::DeviceBVH(VisionarayGlobalState *s) : m_state(s)
{}

template<typename P>
void DeviceBVH<P>::update(const P *prims, unsigned numPrims, unsigned flags)
{
  m_primitives = prims;
  m_numPrimitives = numPrims;
  m_flags = flags;

  m_lastUpdate = newTimeStamp();
}

template<typename P>
typename DeviceBVH<P>::DeviceBVH2::bvh_ref DeviceBVH<P>::deviceBVH2() {
  rebuildDeviceBVH2();
  return m_deviceBVH2.ref();
}

template<typename P>
typename DeviceBVH<P>::DeviceIndexBVH2::bvh_ref DeviceBVH<P>::deviceIndexBVH2() {
  rebuildDeviceIndexBVH2();
  return m_deviceIndexBVH2.ref();
}

template<typename P>
typename DeviceBVH<P>::DeviceBVH4::bvh_ref DeviceBVH<P>::deviceBVH4() {
  rebuildDeviceBVH4();
  return m_deviceBVH4.ref();
}

template<typename P>
TimeStamp DeviceBVH<P>::lastUpdateTime() const {
  return m_lastUpdate;
}

template<typename P>
TimeStamp DeviceBVH<P>::lastRebuildTime() const {
  // this is lazy.. TODO: provide finer granularity
  // only if needed (i.e., the same BVH is built twice,
  // yet with different low-level types, which is rather
  // unlikely as of now..)
  return std::max(m_hostRebuild.BVH2,std::max(
                  m_hostRebuild.IndexBVH2,std::max(
                  m_hostRebuild.BVH4,std::max(
                  m_deviceRebuild.BVH2,std::max(
                  m_deviceRebuild.IndexBVH2,
                  m_deviceRebuild.BVH4)))));
}

template<typename P>
void DeviceBVH<P>::rebuildHostBVH2() {
  if (m_hostRebuild.BVH2 >= m_lastUpdate) {
    return;
  }

  if (m_flags & BVH_FLAG_PREFER_FAST_BUILD) {
    lbvh_builder builder;
    m_hostBVH2 = builder.build(bvh<P>{},
                               m_primitives,
                               m_numPrimitives);
  } else {
    binned_sah_builder builder;
    builder.enable_spatial_splits(m_flags & BVH_FLAG_ENABLE_SPATIAL_SPLITS);
    m_hostBVH2 = builder.build(bvh<P>{},
                               m_primitives,
                               m_numPrimitives);
  }

  m_hostRebuild.BVH2 = newTimeStamp();
}

template<typename P>
void DeviceBVH<P>::rebuildHostIndexBVH2() {
  if (m_hostRebuild.IndexBVH2 >= m_lastUpdate) {
    return;
  }

  P *hPrimitives{nullptr};
#if defined(WITH_CUDA)
  cudaPointerAttributes attributes = {};
  CUDA_SAFE_CALL(cudaPointerGetAttributes(&attributes,m_primitives));
  if (attributes.devicePointer) {
    hPrimitives = (P *)std::malloc(sizeof(P)*m_numPrimitives);
    CUDA_SAFE_CALL(cudaMemcpy(hPrimitives,m_primitives,sizeof(P)*m_numPrimitives,
                              cudaMemcpyDeviceToHost));
  } else {
    hPrimitives = (P *)m_primitives;
  }
#elif defined(WITH_HIP)
  hitPointerAttribute_t attributes = {};
  HIP_SAFE_CALL(hipPointerGetAttributes(&attributes,m_primitives));
  if (attributes.memoryType == hipMemoryTypeDevice) {
    hPrimitives = (P *)std::malloc(sizeof(P)*m_numPrimitives);
    HIP_SAFE_CALL(hipMemcpy(hPrimitives,m_primitives,sizeof(P)*m_numPrimitives,
                            hipMemcpyDeviceToHost));
  }
#else
  hPrimitives = (P *)m_primitives;
#endif

  if (m_flags & BVH_FLAG_PREFER_FAST_BUILD) {
    lbvh_builder builder;
    m_hostIndexBVH2 = builder.build(index_bvh<P>{},
                                    hPrimitives,
                                    m_numPrimitives);
  } else {
    binned_sah_builder builder;
    builder.enable_spatial_splits(m_flags & BVH_FLAG_ENABLE_SPATIAL_SPLITS);
    m_hostIndexBVH2 = builder.build(index_bvh<P>{},
                                    hPrimitives,
                                    m_numPrimitives);
  }

#if defined(WITH_CUDA)
  if (attributes.devicePointer) {
    std::free(hPrimitives);
  }
#elif defined(WITH_HIP)
  if (attributes.memoryType == hipMemoryTypeDevice) {
    std::free(hPrimitives);
  }
#endif

  m_hostRebuild.IndexBVH2 = newTimeStamp();
}

template<typename P>
void DeviceBVH<P>::rebuildHostBVH4() {
  if (m_hostRebuild.BVH4 >= m_lastUpdate) {
    return;
  }

  rebuildHostBVH2();

  bvh_collapser collapser;
  collapser.collapse(m_hostBVH2, m_hostBVH4, deviceState()->threadPool);

  m_hostRebuild.BVH4 = newTimeStamp();
}

template<typename P>
void DeviceBVH<P>::rebuildDeviceBVH2() {
  if (m_deviceRebuild.BVH2 >= m_lastUpdate) {
    return;
  }

  m_deviceRebuild.BVH2 = newTimeStamp();
}

template<typename P>
void DeviceBVH<P>::rebuildDeviceIndexBVH2() {
  if (m_deviceRebuild.IndexBVH2 >= m_lastUpdate) {
    return;
  }

  if (m_flags & BVH_FLAG_PREFER_FAST_BUILD) {
    P *dPrimitives{nullptr};
#if defined(WITH_CUDA)
    cudaPointerAttributes attributes = {};
    CUDA_SAFE_CALL(cudaPointerGetAttributes(&attributes,m_primitives));
    if (attributes.devicePointer) {
      dPrimitives = (P *)m_primitives;
    } else {
      CUDA_SAFE_CALL(cudaMalloc(&dPrimitives,sizeof(P)*m_numPrimitives));
      CUDA_SAFE_CALL(cudaMemcpy(dPrimitives,m_primitives,sizeof(P)*m_numPrimitives,
                                cudaMemcpyHostToDevice));
    }
#elif defined(WITH_HIP)
    hipPointerAttribute_t attributes = {};
    HIP_SAFE_CALL(hipPointerGetAttributes(&attributes,m_primitives));
    if (attributes.memoryType == hipMemoryTypeDevice) {
      dPrimitives = (P *)m_primitives;
    } else {
      HIP_SAFE_CALL(hipMalloc(&dPrimitives,sizeof(P)*m_numPrimitives));
      HIP_SAFE_CALL(hipMemcpy(dPrimitives,m_primitives,sizeof(P)*m_numPrimitives,
                              hipMemcpyHostToDevice));
    }
#else
    dPrimitives = (P *)m_primitives;
#endif
    lbvh_builder builder;
    m_deviceIndexBVH2 = builder.build(DeviceIndexBVH2{},
                                      dPrimitives,
                                      m_numPrimitives);
#if defined(WITH_CUDA)
    if (!attributes.devicePointer) {
      CUDA_SAFE_CALL(cudaFree(dPrimitives));
    }
#elif defined(WITH_HIP)
    if (attributes.memoryType == hipMemoryTypeHost) {
      HIP_SAFE_CALL(hipFree(dPrimitives));
    }
#endif
  } else {
    // Until we have a high-quality GPU builder, do that on the CPU!
    rebuildHostIndexBVH2(); 

    m_deviceIndexBVH2 = DeviceIndexBVH2(m_hostIndexBVH2);
  }

  m_deviceRebuild.IndexBVH2 = newTimeStamp();
}

template<typename P>
void DeviceBVH<P>::rebuildDeviceBVH4() {
  if (m_deviceRebuild.BVH4 >= m_lastUpdate) {
    return;
  }

  rebuildHostBVH4();

  m_deviceBVH4 = m_hostBVH4;

  m_deviceRebuild.BVH4 = newTimeStamp();
}

template<typename P>
VisionarayGlobalState *DeviceBVH<P>::deviceState() {
  return m_state;
}

} // namespace visionaray
