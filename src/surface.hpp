#ifndef _SURFACE_H
#define _SURFACE_H

#include <string>
#include <vector>

#include "mesh.hpp"
#include "vec3.hpp"

using std::vector, std::string;

struct Material {
    bool  collect{false};
    float sticking_coeff{0.0f};
};

struct Emitter {
    bool  emit{false};
    float flux{0.0};
    float velocity{1.0};
    float spread{0.1};
    bool  reverse{false};
};

class Surface {
public:
    string name{"noname"};

    // Emitter options
    Emitter emitter{};

    // Material options
    Material material{};

    // Geometric options
    Mesh      mesh{};
    Transform transform{};
    vec3      color{0.5f, 0.5f, 0.5f};

    Surface()  = default;
    ~Surface() = default;
};

#endif