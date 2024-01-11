
project(anari_library_visionaray LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari 0.7.0 REQUIRED)
find_package(GLEW REQUIRED) # required by visionaray (TODO!)
find_package(visionaray 0.3.4 REQUIRED)

option(ANARI_VISIONARAY_ENABLE_CUDA "Compile the CUDA device" OFF)
set(cuda ${ANARI_VISIONARAY_ENABLE_CUDA})
if (cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

anari_generate_queries(
  NAME visionaray
  PREFIX VisionarayDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
)

if (cuda)
  anari_generate_queries(
    NAME visionaray_cuda
    PREFIX VisionarayDevice
    CPP_NAMESPACE visionaray
    JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_cuda_device.json
    JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

add_library(${PROJECT_NAME} SHARED)
if (cuda)
  add_library(${PROJECT_NAME}_cuda SHARED)
endif()

set(SOURCES
    array/Array1D.cpp
    array/Array2D.cpp
    array/Array3D.cpp
    array/ObjectArray.cpp
    camera/Camera.cpp
    camera/Matrix.cpp
    camera/Perspective.cpp
    renderer/DirectLight.cpp
    renderer/Raycast.cpp
    renderer/Renderer.cpp
    scene/light/Light.cpp
    scene/light/Directional.cpp
    scene/light/HDRI.cpp
    scene/light/Point.cpp
    scene/light/Spot.cpp
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
    scene/volume/spatial_field/GridAccel.cpp
    scene/volume/spatial_field/SpatialField.cpp
    scene/volume/spatial_field/StructuredRegularField.cpp
    scene/volume/TransferFunction1D.cpp
    scene/volume/Volume.cpp
    scene/Group.cpp
    scene/Instance.cpp
    scene/VisionarayScene.cpp
    scene/World.cpp
    Object.cpp
    VisionarayDevice.cpp
    VisionarayDeviceQueries.cpp
    VisionarayGlobalState.cpp
    VisionarayLibrary.cpp
)

target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_sources(${PROJECT_NAME} PRIVATE
  frame/Frame.cpp
  scene/volume/spatial_field/BlockStructuredField.cpp
  scene/volume/spatial_field/UnstructuredField.cpp
)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "VISIONARAY_DEVICE_INTERFACE"
)

target_link_libraries(${PROJECT_NAME}
    visionaray::visionaray anari::anari anari::helium)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (cuda)
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2 -g")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

  target_sources(${PROJECT_NAME}_cuda PRIVATE ${SOURCES})
  target_sources(${PROJECT_NAME}_cuda PRIVATE
    frame/Frame.cu
    scene/volume/spatial_field/BlockStructuredField.cu
    scene/volume/spatial_field/UnstructuredField.cu
    scene/VisionaraySceneGPU.cu
  )
  set_source_files_properties(
    frame/Frame.cu
    scene/volume/spatial_field/BlockStructuredField.cu
    scene/volume/spatial_field/UnstructuredField.cu
    scene/VisionaraySceneGPU.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
  )
  target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE WITH_CUDA=1)

  generate_export_header(${PROJECT_NAME}_cuda
    EXPORT_MACRO_NAME "VISIONARAY_DEVICE_INTERFACE"
  )

  target_link_libraries(${PROJECT_NAME}_cuda
      visionaray::visionaray anari::anari anari::helium)

  if (TARGET CUDA::toolkit)
    target_link_libraries(${PROJECT_NAME}_cuda CUDA::toolkit)
  endif()

  target_include_directories(${PROJECT_NAME}_cuda PUBLIC
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
