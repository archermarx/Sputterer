cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Sputterer LANGUAGES C CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw)
add_subdirectory(tomlplusplus)

find_package(OpenGL REQUIRED)

if (OPENGL_FOUND)
    message("opengl found")
    message("include dir: ${OPENGL_INCLUDE_DIR}")
    message("link libraries: ${OPENGL_gl_LIBRARY}")
else (OPENGL_FOUND)
    message("opengl not found")
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/glad/include
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
        "${PROJECT_SOURCE_DIR}/include/Faddeeva.cpp"
        "${PROJECT_SOURCE_DIR}/glad/src/*.c"
)

add_executable(sputterer ${sources})
target_link_libraries(sputterer PRIVATE tomlplusplus::tomlplusplus glfw ${OPENGL_gl_LIBRARY})
target_compile_features(sputterer PUBLIC cxx_std_20)

set_property(TARGET sputterer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# detect cxx compiler and set flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -O2 -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "/Wall /O2 /EHsc")
endif()

set(CMAKE_CUDA_FLAGS "-expt-relaxed-constexpr -Xcudafe=\"--diag_suppress=20012\"")
