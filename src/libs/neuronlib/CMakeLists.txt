add_library(neuronlib
    neuron.cpp
    spatial.cpp
    spatial_operations.cpp
    geometry.cpp
    flow_field.cpp
    dendrite.cpp
)

target_include_directories(neuronlib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(neuronlib PUBLIC cxx_std_17)