#pragma once
#ifndef SURFACE_HPP
#define SURFACE_HPP

#include <string>
#include <vector>

#include "mesh.hpp"
#include "vec3.hpp"

using std::vector, std::string;

struct Material {
    bool  collect{false};
    float sticking_coeff{0.0f};
    float diffuse_coeff{0.0f};
    float temperature_K{300.0f};
};

struct Emitter {
    bool  emit{false};
    float flux{0.0};
    float velocity{1.0};
    float spread{0.1};
    bool  reverse{false};
};

struct Surface {
    // Name of surface
    string name{"noname"};

    // Emitter options
    Emitter emitter{};

    // Material options
    Material material{};

    // Geometric options
    Mesh      mesh{};
    Transform transform{};
    vec3      color{0.5f, 0.5f, 0.5f};
};

#endif