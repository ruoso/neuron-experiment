find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50f33f9142461b5f4b6a47a30b4e5c.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_neuronlib
    test_spatial_operations.cpp
)

target_link_libraries(test_neuronlib
    neuronlib
    gtest_main
)

target_include_directories(test_neuronlib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(test_neuronlib)