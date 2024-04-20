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
    bool   emit{false};
    bool   collect{false};

    Mesh mesh{};
    vec3 scale;
    vec3 translate;
    vec3 color;

    Surface()  = default;
    ~Surface() = default;
    Surface(string name, bool emit, bool collect, vec3 scale, vec3 translate, vec3 color);
};

#endif