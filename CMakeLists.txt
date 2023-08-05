
project(anari_library_visionaray LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari 0.4.0 REQUIRED)
find_package(GLEW REQUIRED) # required by visionaray (TODO!)
find_package(visionaray REQUIRED)

anari_generate_queries(
  NAME visionaray
  PREFIX VisionarayDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_device.json
)

anari_generate_debug_objects(
  NAME visionaray
  PREFIX VisionarayDevice
  CPP_NAMESPACE visionaray
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/visionaray_device.json
)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE
    array/Array.cpp
    array/Array1D.cpp
    array/Array2D.cpp
    array/Array3D.cpp
    array/ObjectArray.cpp
    camera/Camera.cpp
    camera/Perspective.cpp
    frame/Frame.cpp
    renderer/Renderer.cpp
    scene/light/Light.cpp
    scene/surface/geometry/Geometry.cpp
    scene/surface/geometry/Triangle.cpp
    scene/surface/material/sampler/Sampler.cpp
    scene/surface/material/Material.cpp
    scene/surface/material/Matte.cpp
    scene/surface/Surface.cpp
    scene/volume/spatial_field/SpatialField.cpp
    scene/volume/Volume.cpp
    scene/Group.cpp
    scene/Instance.cpp
    scene/VisionarayScene.cpp
    scene/World.cpp
    Object.cpp
    VisionarayDevice.cpp
    VisionarayDeviceDebugFactory.cpp
    VisionarayDeviceQueries.cpp
    VisionarayGlobalState.cpp
    VisionarayLibrary.cpp
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC visionaray::visionaray anari::anari anari::helium anari::anari_library_debug)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)
