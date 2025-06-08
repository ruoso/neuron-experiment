cmake_minimum_required(VERSION 3.20)
project(neuron-experiment VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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