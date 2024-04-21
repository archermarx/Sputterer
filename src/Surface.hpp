#ifndef _SURFACE_H
#define _SURFACE_H

#include <string>
#include <vector>

#include "Mesh.hpp"
#include "Vec3.hpp"

using std::vector, std::string;

class Surface {
public:
    string name{"noname"};

    // Emitter options
    bool  emit{false};
    float emitter_flux{0.0};

    // Collector options
    bool collect{false};

    Mesh mesh{};
    vec3 scale{1.0f};
    vec3 translate{0.0f};
    vec3 color{0.5f, 0.5f, 0.5f};

    Surface()  = default;
    ~Surface() = default;
    Surface(string name, bool emit, bool collect, vec3 scale, vec3 translate, vec3 color);
};

#endif