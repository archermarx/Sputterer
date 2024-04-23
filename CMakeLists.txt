cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Sputterer LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

find_package(glfw3 3.3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/imgui
)

file(GLOB sources
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/include/*.cuh"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/imgui/*.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/*.h"
    "${PROJECT_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp"
    "${PROJECT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp"
    "${PROJECT_SOURCE_DIR}/cpp/imgui_stdlib.*"
)

add_executable(sputterer ${sources})
target_link_libraries(sputterer PRIVATE tomlplusplus::tomlplusplus glfw)
target_compile_features(sputterer PUBLIC cxx_std_20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O2 -g")
set(CMAKE_CUDA_FLAGS "-arch=sm_61 -std=c++20 -expt-relaxed-constexpr -Xcudafe=\"--diag_suppress=20012\"")