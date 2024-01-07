
project(anari_library_visionaray LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari 0.7.0 REQUIRED)
find_package(GLEW REQUIRED) # required by visionaray (TODO!)
find_package(visionaray 0.3.4 REQUIRED)

anari_generate_queries(
  NAME visionaray
  PREFIX VisionarayDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}
)

option(ANARI_VISIONARAY_ENABLE_CUDA "Compile with CUDA" OFF)
set(cuda ${ANARI_VISIONARAY_ENABLE_CUDA})
if (cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE
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
    scene/volume/spatial_field/BlockStructuredField.cpp
    scene/volume/spatial_field/GridAccel.cpp
    scene/volume/spatial_field/SpatialField.cpp
    scene/volume/spatial_field/StructuredRegularField.cpp
    scene/volume/spatial_field/UnstructuredField.cpp
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

if (cuda)
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O2 -g")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

  target_sources(${PROJECT_NAME} PRIVATE
    frame/Frame.cu
    scene/VisionaraySceneGPU.cu
  )
  set_source_files_properties(frame/Frame.cu scene/VisionaraySceneGPU.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA=1)
else()
  target_sources(${PROJECT_NAME} PRIVATE frame/Frame.cpp)
endif()

target_link_libraries(${PROJECT_NAME}
    visionaray::visionaray anari::anari anari::helium)
if (TARGET CUDA::toolkit)
  target_link_libraries(${PROJECT_NAME} CUDA::toolkit)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

## Installation ##

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
