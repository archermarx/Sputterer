#ifndef _SURFACE_H
#define _SURFACE_H

#include <string>
#include <vector>

#include "mesh.hpp"
#include "vec3.hpp"

using std::vector, std::string;

class Surface {
public:
    string name{"noname"};

    // Emitter options
    bool  emit{false};
    float emitter_flux{0.0};

    // Collector options
    bool collect{false};

    Mesh      mesh{};
    Transform transform{};
    vec3      color{0.5f, 0.5f, 0.5f};

    Surface()  = default;
    ~Surface() = default;
    Surface(string name, bool emit, bool collect);
};

#endif