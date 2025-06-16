find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

# Create an OBJECT library for reusable experiment components
add_library(experiment-components OBJECT
    neural_simulation.cpp
    brain_visualization.cpp
    visualization.cpp
)

target_link_libraries(experiment-components
    neuronlib
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(experiment-components PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(experiment-components PRIVATE ${SDL2_CFLAGS_OTHER})

# Create the grid experiment executable
add_executable(neuron-grid-experiment
    neuron_grid_experiment.cpp
)

target_link_libraries(neuron-grid-experiment
    neuronlib
    experiment-components
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(neuron-grid-experiment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(neuron-grid-experiment PRIVATE ${SDL2_CFLAGS_OTHER})