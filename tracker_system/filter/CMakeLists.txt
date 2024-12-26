cmake_minimum_required(VERSION 3.14)
project(filter VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true --expt-extended-lambda -Xcompiler=-Wno-unknown-pragmas")

message(STATUS "----------------------Config Filter Starting-----------------")

# Enable Debug macro for Debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    message(STATUS "Filter Debug build: DEBUG macro is defined")
else()
    # Add Release-specific optimizations
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
    message(STATUS "Filter Release build: Optimizations enabled: --use_fast_math -O3")
endif()

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 87 89 90)

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories for the filter
set(FILTER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(FILTER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ROOT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Create the filter library
add_library(filter STATIC)

target_sources(filter
    PRIVATE
        ${FILTER_SRC_DIR}/particle_states.cu
        ${FILTER_SRC_DIR}/kernels.cu
        ${FILTER_SRC_DIR}/particle_filter.cpp
    PUBLIC
        $<BUILD_INTERFACE:${FILTER_INCLUDE_DIR}/filter/particle_states.cuh>
        $<BUILD_INTERFACE:${FILTER_INCLUDE_DIR}/filter/kernels.cuh>
        $<BUILD_INTERFACE:${FILTER_INCLUDE_DIR}/filter/particle_filter.hpp>
        $<BUILD_INTERFACE:${ROOT_INCLUDE_DIR}/filter_interface.hpp>
)

target_include_directories(filter
    PUBLIC
        $<BUILD_INTERFACE:${FILTER_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${ROOT_INCLUDE_DIR}>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
)

# Enable CUDA separable compilation
set_target_properties(filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(filter
    PUBLIC
        CUDA::cublas
        CUDA::cusolver
        CUDA::curand
        CUDA::cudart
        ${OpenCV_LIBS}
)

# Example application
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR FILTER_BUILD_EXAMPLES)
    message(STATUS "Building Filter Example!")
    add_executable(filter_example examples/filter_example.cu)

    target_include_directories(filter_example
        PRIVATE
            ${FILTER_INCLUDE_DIR}
            ${ROOT_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(filter_example
        PRIVATE
            filter
            ${OpenCV_LIBS}
            CUDA::cudart
    )
endif()

install(TARGETS filter EXPORT filterTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ${FILTER_INCLUDE_DIR}/ DESTINATION include)
install(DIRECTORY ${ROOT_INCLUDE_DIR}/ DESTINATION include)

export(EXPORT filterTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/filterTargets.cmake
    NAMESPACE Filter::
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/filterConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/filterConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/filterConfig.cmake"
    INSTALL_DESTINATION lib/cmake/filter
)

install(EXPORT filterTargets
    FILE filterTargets.cmake
    NAMESPACE Filter::
    DESTINATION lib/cmake/filter
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/filterConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/filterConfigVersion.cmake"
    DESTINATION lib/cmake/filter
)

message(STATUS "----------------------Config Filter Done!!!-----------------")
