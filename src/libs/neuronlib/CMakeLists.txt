add_library(neuronlib
    neuron.cpp
    spatial.cpp
    spatial_operations.cpp
    geometry.cpp
    flow_field.cpp
    dendrite.cpp
    brain.cpp
    activation.cpp
    sensor.cpp
    actuator.cpp
)

target_include_directories(neuronlib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(neuronlib PUBLIC
    spdlog::spdlog
)

target_compile_features(neuronlib PUBLIC cxx_std_17)