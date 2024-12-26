cmake_minimum_required(VERSION 3.14)
project(tracker_system VERSION 1.0.0 LANGUAGES CXX CUDA)

# General Build Configuration
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable Debug macro for Debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "Debug build: DEBUG macro is defined")
else()
    message(STATUS "Release build: DEBUG macro is not defined")
endif()

add_compile_definitions(TRITON_ENABLE_GPU)

# Options to build examples
option(PERCEPTION_BUILD_EXAMPLES "Build perception examples" ON)
option(BYTETRACKER_BUILD_EXAMPLES "Build bytetrack examples" ON)
option(FILTER_BUILD_EXAMPLES "Build filter examples" ON)

# CUDA Configuration
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 87 89 90)
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-unknown-pragmas")
set(CMAKE_CUDA_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> -dlink <LINK_FLAGS> <OBJECTS> -o <TARGET> -lcudadevrt -lcudart_static -lrt -lpthread -ldl")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

# Triton Client Configuration
set(TritonClientInstall_DIR "/opt/triton-client/build/install" CACHE PATH "Path to Triton Client installation")
set(TritonClientLib_DIR "${TritonClientInstall_DIR}/lib")
include_directories(${TritonClientInstall_DIR}/include)

# OpenCV Configuration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DCUDA_NO_HOST_MEM)

# Include Directories
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/filter/include
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/include
)

# Add Subdirectories
add_subdirectory(perception)
add_subdirectory(bytetrack)
add_subdirectory(filter)

# Tracker System Library
add_library(tracker_system STATIC src/tracker_system.cpp)

target_include_directories(tracker_system PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(tracker_system PUBLIC
    perception
    bytetrack
    filter
    ${TritonClientLib_DIR}/libgrpcclient.so
)

set_target_properties(tracker_system PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Example Applications
if (PERCEPTION_BUILD_EXAMPLES OR BYTETRACKER_BUILD_EXAMPLES OR FILTER_BUILD_EXAMPLES)
    message(STATUS "Building example applications.")
    
    # List of example executables
    set(EXAMPLES example_system)

    foreach(EXAMPLE ${EXAMPLES})
        add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)

        target_include_directories(${EXAMPLE} PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${TritonClientInstall_DIR}/include
        )

        target_link_libraries(${EXAMPLE} PRIVATE
            tracker_system
            ${TritonClientLib_DIR}/libgrpcclient.so
            ${OpenCV_LIBS}
        )

        set_target_properties(${EXAMPLE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach()
endif()