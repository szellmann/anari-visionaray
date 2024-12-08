
cmake_minimum_required(VERSION 3.23)

set(CMAKE_BUILD_TYPE_INIT Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Establish project

project(anari_library_visionaray LANGUAGES C CXX)

include(GNUInstallDirs)

# build external libraries

add_subdirectory(external)

# note: we're often tracking the most recent changes from ANARI-SDK
# (on branch "next_release")
find_package(anari 0.10.0 REQUIRED)
find_package(visionaray 0.4.2 REQUIRED)

option(ANARI_VISIONARAY_ENABLE_CUDA "Compile the CUDA device" OFF)
set(cuda ${ANARI_VISIONARAY_ENABLE_CUDA})
if (cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

option(ANARI_VISIONARAY_ENABLE_HIP "Compile the HIP device" OFF)
set(hip ${ANARI_VISIONARAY_ENABLE_HIP})
if (hip)
  enable_language(HIP)
  find_package(hip)
endif()

option(ANARI_VISIONARAY_ENABLE_NANOVDB "Enable NanoVDB spatial field type" OFF)
set(nanovdb ${ANARI_VISIONARAY_ENABLE_NANOVDB})

anari_generate_queries(
  NAME visionaray
  PREFIX VisionarayDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
)

anari_generate_queries(
  NAME visionaray_cuda
  PREFIX VisionarayCUDADevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_cuda_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
)

anari_generate_queries(
  NAME visionaray_hip
  PREFIX VisionarayHIPDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_hip_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(${PROJECT_NAME} SHARED)
if (cuda)
  add_library(${PROJECT_NAME}_cuda SHARED)
endif()
if (hip)
  add_library(${PROJECT_NAME}_hip SHARED)
endif()

set(SOURCES
    camera/Camera.cpp
    camera/Matrix.cpp
    camera/Orthographic.cpp
    camera/Perspective.cpp
    renderer/DirectLight.cpp
    renderer/Raycast.cpp
    renderer/Renderer.cpp
    scene/light/Light.cpp
    scene/light/Directional.cpp
    scene/light/HDRI.cpp
    scene/light/Point.cpp
    scene/light/Quad.cpp
    scene/light/Spot.cpp
    scene/surface/geometry/BezierCurve.cpp
    scene/surface/geometry/Cone.cpp
    scene/surface/geometry/Cylinder.cpp
    scene/surface/geometry/Geometry.cpp
    scene/surface/geometry/ISOSurface.cpp
    scene/surface/geometry/Quad.cpp
    scene/surface/geometry/Sphere.cpp
    scene/surface/geometry/Triangle.cpp
    scene/surface/material/sampler/Image1D.cpp
    scene/surface/material/sampler/Image2D.cpp
    scene/surface/material/sampler/Image3D.cpp
    scene/surface/material/sampler/PrimitiveSampler.cpp
    scene/surface/material/sampler/Sampler.cpp
    scene/surface/material/sampler/TransformSampler.cpp
    scene/surface/material/Material.cpp
    scene/surface/material/Matte.cpp
    scene/surface/material/PBM.cpp
    scene/surface/Surface.cpp
    scene/volume/spatial_field/SpatialField.cpp
    scene/volume/spatial_field/StructuredRegularField.cpp
    scene/volume/TransferFunction1D.cpp
    scene/volume/Volume.cpp
    scene/Group.cpp
    scene/Instance.cpp
    scene/MotionTransform.cpp
    scene/VisionarayScene.cpp
    scene/World.cpp
    Object.cpp
    VisionarayDevice.cpp
    VisionarayGlobalState.cpp
    VisionarayLibrary.cpp
)

target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_sources(${PROJECT_NAME} PRIVATE
  frame/Frame.cpp
  renderer/DirectLight_impl.cpp
  renderer/Raycast_impl.cpp
  scene/volume/spatial_field/BlockStructuredField.cpp
  scene/volume/spatial_field/GridAccel.cpp
  scene/volume/spatial_field/UnstructuredField.cpp
  VisionarayDeviceQueries.cpp
)

if (nanovdb)
  target_sources(${PROJECT_NAME} PRIVATE scene/volume/spatial_field/NanoVDBField.cpp)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_NANOVDB=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:vsnray_nanovdb>)
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "VISIONARAY_DEVICE_INTERFACE"
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    visionaray::visionaray anari::anari anari::helium)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (cuda)
# set_target_properties(${PROJECT_NAME}_cuda PROPERTIES CUDA_ARCHITECTURES "all-major")
  set_target_properties(${PROJECT_NAME}_cuda PROPERTIES CUDA_ARCHITECTURES "native")

  target_sources(${PROJECT_NAME}_cuda PRIVATE ${SOURCES})
  target_sources(${PROJECT_NAME}_cuda PRIVATE
    frame/Frame.cu
    renderer/DirectLight_impl.cu
    renderer/Raycast_impl.cu
    scene/volume/spatial_field/BlockStructuredField.cu
    scene/volume/spatial_field/GridAccel.cu
    scene/volume/spatial_field/UnstructuredField.cu
    scene/VisionaraySceneGPU.cu
    VisionarayCUDADeviceQueries.cpp
  )
  set_source_files_properties(
    frame/Frame.cu
    renderer/DirectLight_impl.cu
    renderer/Raycast_impl.cu
    scene/volume/spatial_field/BlockStructuredField.cu
    scene/volume/spatial_field/GridAccel.cu
    scene/volume/spatial_field/UnstructuredField.cu
    scene/VisionaraySceneGPU.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
  )
  target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE WITH_CUDA=1)
  if (nanovdb)
    target_sources(${PROJECT_NAME}_cuda PRIVATE
        scene/volume/spatial_field/NanoVDBField.cu)
    set_source_files_properties(
      scene/volume/spatial_field/NanoVDBField.cu
      PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
    )
    target_link_libraries(${PROJECT_NAME}_cuda PRIVATE
      $<BUILD_INTERFACE:vsnray_nanovdb>)
    target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE WITH_NANOVDB=1)
  endif()

  generate_export_header(${PROJECT_NAME}_cuda
    EXPORT_MACRO_NAME "VISIONARAY_DEVICE_INTERFACE"
  )

  target_link_libraries(${PROJECT_NAME}_cuda PUBLIC
      visionaray::visionaray anari::anari anari::helium)

  if (TARGET CUDA::cudart)
    target_link_libraries(${PROJECT_NAME}_cuda PUBLIC CUDA::cudart)
  endif()

  target_include_directories(${PROJECT_NAME}_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
endif()

if (hip)
  target_sources(${PROJECT_NAME}_hip PRIVATE ${SOURCES})
  target_sources(${PROJECT_NAME}_hip PRIVATE
    frame/Frame.hip
    renderer/DirectLight_impl.hip
    renderer/Raycast_impl.hip
    scene/VisionaraySceneGPU.hip
    VisionarayHIPDeviceQueries.cpp
  )
  target_compile_definitions(${PROJECT_NAME}_hip PRIVATE WITH_HIP=1)
  if (nanovdb)
    message(WARNING "No VDB support with HIP")
  endif()

  target_link_libraries(${PROJECT_NAME}_hip
      visionaray::visionaray anari::anari anari::helium)

  if (TARGET hip::host)
    target_link_libraries(${PROJECT_NAME}_hip hip::host)
  endif()

  generate_export_header(${PROJECT_NAME}_hip
    EXPORT_MACRO_NAME "VISIONARAY_DEVICE_INTERFACE"
  )

  target_include_directories(${PROJECT_NAME}_hip PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
endif()

## Installation ##

set(DEVICE_LIBS ${PROJECT_NAME})
if (cuda)
  set(DEVICE_LIBS ${DEVICE_LIBS} ${PROJECT_NAME}_cuda)
endif()

install(TARGETS ${DEVICE_LIBS}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
