cmake_minimum_required(VERSION 3.20)

project(
    VectoRex
    VERSION 0.1.0
    DESCRIPTION "A vector database in C++"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(VectoRex
    src/main.cpp
    src/vector.cpp
    src/distance.cpp
    src/vector_store.cpp
)

target_include_directories(VectoRex
    PRIVATE
        include
)

enable_testing()

add_executable(VectoRexVectorTests
    tests/vector_test.cpp
    src/vector.cpp
)

target_include_directories(VectoRexVectorTests
    PRIVATE
        include
)

add_test(
    NAME VectorTest
    COMMAND VectoRexVectorTests
)

add_executable(VectoRexDistanceTests
    tests/distance_test.cpp
    src/vector.cpp
    src/distance.cpp
)

target_include_directories(VectoRexDistanceTests
    PRIVATE
        include
)

add_test(
    NAME DistanceTests
    COMMAND VectoRexDistanceTests
)
