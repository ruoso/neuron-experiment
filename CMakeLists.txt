cmake_minimum_required(VERSION 3.20)
project(neuron-experiment VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA globally
find_package(CUDAToolkit REQUIRED)

# Add spdlog for logging
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src/libs/neuronlib)
add_subdirectory(src/app)

enable_testing()
add_subdirectory(tests)

add_custom_target(lint
    COMMAND cpplint --filter=-legal/copyright src/**/*.cpp include/**/*.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running cpplint"
)