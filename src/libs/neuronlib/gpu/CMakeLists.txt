# GPU acceleration module for neuronlib
cmake_minimum_required(VERSION 3.18)



# Set CUDA architecture - adjust based on target GPU
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing, Ampere, Ada Lovelace

# GPU library
add_library(neuronlib_gpu STATIC
    gpu_interface.h
    gpu_converter.h
    gpu_converter.cpp
    gpu_kernel.cu
    gpu_implementation.cu
)

# Set CUDA properties
set_target_properties(neuronlib_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(neuronlib_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link CUDA libraries
target_link_libraries(neuronlib_gpu
    CUDA::cudart
    CUDA::cuda_driver
)


# Compiler-specific options
target_compile_options(neuronlib_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -O3
        -Xcompiler -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

# Add dependencies (spdlog should be available from parent)
if(TARGET spdlog::spdlog)
    target_link_libraries(neuronlib_gpu spdlog::spdlog)
endif()

# Export for parent CMakeLists.txt
set(NEURONLIB_GPU_TARGET neuronlib_gpu PARENT_SCOPE)