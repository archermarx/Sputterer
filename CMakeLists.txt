cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Sputterer LANGUAGES CXX CUDA)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

file(GLOB sources
    "${PROJECT_SOURCE_DIR}/include/*.cuh"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)

add_executable(sputterer ${sources})
target_compile_features(sputterer PUBLIC cxx_std_20)
