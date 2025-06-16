# GPU acceleration module for neuronlib
cmake_minimum_required(VERSION 3.18)

# Convert OpenCL kernel to C string header
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_kernel_source.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernel_header.py 
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernel.cl 
            ${CMAKE_CURRENT_BINARY_DIR}/gpu_kernel_source.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernel.cl
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernel_header.py
    COMMENT "Generating GPU kernel source header"
)

# GPU library
add_library(neuronlib_gpu STATIC
    gpu_interface.h
    gpu_converter.h
    gpu_converter.cpp
    gpu_implementation.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_kernel_source.h
)

# Include directories
target_include_directories(neuronlib_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_include_directories(neuronlib_gpu PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
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