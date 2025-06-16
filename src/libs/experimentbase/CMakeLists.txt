add_library(experimentbase STATIC
    neural_simulation.cpp
    brain_visualization.cpp
    visualization.cpp
)

target_link_libraries(experimentbase
    neuronlib
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(experimentbase PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(experimentbase PRIVATE ${SDL2_CFLAGS_OTHER})

