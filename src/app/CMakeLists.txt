# Create the grid experiment executable
add_executable(neuron-grid-experiment
    neuron_grid_experiment.cpp
)

target_link_libraries(neuron-grid-experiment
    neuronlib
    experimentbase
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(neuron-grid-experiment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(neuron-grid-experiment PRIVATE ${SDL2_CFLAGS_OTHER})