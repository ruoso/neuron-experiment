add_executable(neuron-experiment
    main.cpp
)

target_link_libraries(neuron-experiment
    neuronlib
)

target_include_directories(neuron-experiment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)