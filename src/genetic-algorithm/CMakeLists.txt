cmake_minimum_required(VERSION 3.20)

# Create genetic algorithm executable
add_executable(genetic-algorithm
    main.cpp
    genetic_algorithm.cpp
)

target_include_directories(genetic-algorithm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(genetic-algorithm PRIVATE
    spdlog::spdlog
)

# Ensure creature-experiment is built before genetic-algorithm
add_dependencies(genetic-algorithm creature-experiment)

# Set output directory
set_target_properties(genetic-algorithm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)