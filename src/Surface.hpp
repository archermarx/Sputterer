#ifndef _SURFACE_H
#define _SURFACE_H

#include <string>
#include <vector>

#include "Mesh.hpp"
#include "Shader.hpp"

using std::vector, std::string;

class Surface {
public:
    string name{"noname"};
    bool emit{false};
    bool collect{false};

    Mesh mesh{};
    glm::vec3 scale;
    glm::vec3 translate;
    glm::vec3 color;

    Surface()  = default;
    ~Surface() = default;
    Surface(string name, bool emit, bool collect, glm::vec3 scale, glm::vec3 translate, glm::vec3 color);
};

#endif