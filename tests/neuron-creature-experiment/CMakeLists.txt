find_package(GTest REQUIRED)

add_executable(test_world
    test_world.cpp
)

target_link_libraries(test_world
    neuron-creature-experiment
    neuronlib
    GTest::gtest
    GTest::gtest_main
    spdlog::spdlog
)

add_test(NAME WorldTest COMMAND test_world)