cmake_minimum_required(VERSION 3.14)
project(bytetrack VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "----------------------Config ByteTracker Starting-----------------")

# Find Eigen3 library
find_package(Eigen3 REQUIRED)

# Find OpenCV library
find_package(OpenCV REQUIRED)

# Set TensorRT paths
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIB_DIR})
find_library(NVPARSERS_LIB nvonnxparser HINTS ${TENSORRT_LIB_DIR})
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIB_DIR})

if (NOT NVINFER_LIB OR NOT NVPARSERS_LIB OR NOT NVINFER_PLUGIN_LIB)
    message(FATAL_ERROR "TensorRT libraries not found! Make sure TensorRT is installed correctly.")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    # Add Release-specific optimizations
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
    message(STATUS "ByteTrack Release build: Optimizations enabled: --use_fast_math -O3")
endif()

# Include directories for the library
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include # Adding the tracker interface directory
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Define ByteTrack source files
set(BYTE_TRACK_SRCS
    src/BYTETracker.cpp
    src/STrack.cpp
    src/kalmanFilter.cpp
    src/utils.cpp
    src/lapjv.cpp
)

# Define ByteTrack header files
set(BYTE_TRACK_HDRS
    include/BYTETracker.h
    include/STrack.h
    include/kalmanFilter.h
    include/lapjv.h
    include/dataType.h
    include/logging.h
)

# Create the library
add_library(bytetrack STATIC ${BYTE_TRACK_SRCS} ${BYTE_TRACK_HDRS})

# Include directories for the library
target_include_directories(bytetrack
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> # Ensure tracker_interface.hpp is available
        $<INSTALL_INTERFACE:include>
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
)

# Link necessary libraries
target_link_libraries(bytetrack
    PUBLIC
        ${NVINFER_LIB}
        ${NVPARSERS_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
)

# Installation
install(TARGETS bytetrack EXPORT byteTrackTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Export the library for use in other projects
export(EXPORT byteTrackTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/byteTrackTargets.cmake
    NAMESPACE ByteTrack::
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/byteTrackConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/byteTrackConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/byteTrackConfig.cmake"
    INSTALL_DESTINATION lib/cmake/bytetrack
)

install(EXPORT byteTrackTargets
    FILE byteTrackTargets.cmake
    NAMESPACE ByteTrack::
    DESTINATION lib/cmake/bytetrack
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/byteTrackConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/byteTrackConfigVersion.cmake"
    DESTINATION lib/cmake/bytetrack
)

# Example application
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BYTETRACKER_BUILD_EXAMPLES)
    message(STATUS "Building ByteTracker Example!")
    # Add example executable
    add_executable(example_tracks examples/example_tracks.cpp)

    # Link the ByteTrack library with the example executable
    target_link_libraries(example_tracks
        PRIVATE
            bytetrack
            ${OpenCV_LIBS}
            ${NVINFER_LIB}
            ${NVPARSERS_LIB}
            ${NVINFER_PLUGIN_LIB}
            ${CUDA_LIBRARIES}
    )

    # Include directories for the example
    target_include_directories(example_tracks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../include # Ensure tracker_interface.hpp is available
            ${EIGEN3_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
            ${TENSORRT_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
    )
endif()

message(STATUS "----------------------Config ByteTracker Done!!!-----------------")
