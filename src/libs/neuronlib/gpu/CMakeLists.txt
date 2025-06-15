# GPU acceleration module for neuronlib
cmake_minimum_required(VERSION 3.18)

# GPU library
add_library(neuronlib_gpu STATIC
    gpu_interface.h
    gpu_converter.h
    gpu_converter.cpp
    gpu_kernel.cl
    gpu_implementation.cpp
)

# Include directories
target_include_directories(neuronlib_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link OpenCL libraries
target_link_libraries(neuronlib_gpu
    OpenCL::OpenCL
)

# Compiler-specific options
target_compile_options(neuronlib_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

# Set OpenCL target version
target_compile_definitions(neuronlib_gpu PRIVATE CL_TARGET_OPENCL_VERSION=300)

# Add dependencies (spdlog should be available from parent)
if(TARGET spdlog::spdlog)
    target_link_libraries(neuronlib_gpu spdlog::spdlog)
endif()

# Export for parent CMakeLists.txt
set(NEURONLIB_GPU_TARGET neuronlib_gpu PARENT_SCOPE)