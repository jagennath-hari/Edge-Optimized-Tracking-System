cmake_minimum_required(VERSION 3.14)
project(perception VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "----------------------Config Perception Starting-----------------")

# Enable Debug macro for Debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "Perception Debug build: DEBUG macro is defined")
else()
    # Add Release-specific optimizations
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
    message(STATUS "Perception Release build: Optimizations enabled: --use_fast_math -O3")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Debug: Print OpenCV details
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Triton Client directories
set(TritonClientInstall_DIR "/opt/triton-client/build/install")
set(TritonClientLib_DIR "${TritonClientInstall_DIR}/lib")

# Include Triton Client headers
include_directories(${TritonClientInstall_DIR}/include)

# Add the source and include directories
set(PERCEPTION_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PERCEPTION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ROOT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include") # Relative to the perception directory

# Create the perception library
add_library(perception STATIC)

# Add source files
target_sources(perception
    PRIVATE
        ${PERCEPTION_SRC_DIR}/engine.cpp
    PUBLIC
        $<BUILD_INTERFACE:${PERCEPTION_INCLUDE_DIR}/perception/engine.hpp>
)

# Include directories for the library
target_include_directories(perception
    PUBLIC
        $<BUILD_INTERFACE:${PERCEPTION_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${ROOT_INCLUDE_DIR}> # Fixed to use BUILD_INTERFACE for root include
)

# Link libraries
target_link_libraries(perception
    PUBLIC
        ${OpenCV_LIBS}
        ${TritonClientLib_DIR}/libgrpcclient.so
)

# Example application
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR PERCEPTION_BUILD_EXAMPLES)
    message(STATUS "Building Perception Example!")
    add_executable(perception_dataset examples/perception_dataset.cpp)

    target_include_directories(perception_dataset
        PRIVATE
            ${PERCEPTION_INCLUDE_DIR}
            ${ROOT_INCLUDE_DIR} 
            ${OpenCV_INCLUDE_DIRS}
            ${TritonClientInstall_DIR}/include
    )

    target_link_libraries(perception_dataset
        PRIVATE
            perception
            ${OpenCV_LIBS}
            ${TritonClientLib_DIR}/libgrpcclient.so
    )
endif()

# Installation for the library
install(TARGETS perception EXPORT perceptionTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ${PERCEPTION_INCLUDE_DIR}/
    DESTINATION include
)

# Export the perception library
export(EXPORT perceptionTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/perceptionTargets.cmake
    NAMESPACE Perception::
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/perceptionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perceptionConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/perceptionConfig.cmake"
    INSTALL_DESTINATION lib/cmake/perception
)

install(EXPORT perceptionTargets
    FILE perceptionTargets.cmake
    NAMESPACE Perception::
    DESTINATION lib/cmake/perception
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/perceptionConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/perceptionConfigVersion.cmake"
    DESTINATION lib/cmake/perception
)

message(STATUS "----------------------Config Perception Done!!!-----------------")
