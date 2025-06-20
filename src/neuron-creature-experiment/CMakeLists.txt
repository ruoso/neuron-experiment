# Create the creature experiment library
add_library(neuron-creature-experiment
    world_types.cpp
    world.cpp
    creature.cpp
)

target_include_directories(neuron-creature-experiment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(neuron-creature-experiment
    neuronlib
    spdlog::spdlog
)

# Create the creature experiment executable
add_executable(creature-experiment
    creature_experiment.cpp
)

target_link_libraries(creature-experiment
    neuron-creature-experiment
    neuronlib
    experimentbase
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(creature-experiment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(creature-experiment PRIVATE ${SDL2_CFLAGS_OTHER})