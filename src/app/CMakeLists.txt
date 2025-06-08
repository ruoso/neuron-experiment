find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

add_executable(neuron-experiment
    main.cpp
)

target_link_libraries(neuron-experiment
    neuronlib
    ${SDL2_LIBRARIES}
    spdlog::spdlog
)

target_include_directories(neuron-experiment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_compile_options(neuron-experiment PRIVATE ${SDL2_CFLAGS_OTHER})