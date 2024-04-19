#ifndef _MESH_HPP
#define _MESH_HPP

// Standard libraries
#include <cstddef>
#include <string>
#include <vector>

// GLM types
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"

struct Vertex {
    glm::vec3 pos;
    glm::vec3 norm;
};

std::ostream &operator<< (std::ostream &os, const Vertex &v);

struct TriElement {
    unsigned int i1, i2, i3;
};

std::ostream &operator<< (std::ostream &os, const TriElement &t);

struct Transform {
    glm::vec3 color{0.3, 0.3, 0.3};
    glm::vec3 scale{1.0};
    glm::vec3 translate{0.0, 0.0, 0.0};
};

class Mesh {
public:
    size_t numVertices{0};
    size_t numTriangles{0};

    bool smooth{false};

    std::vector<Vertex>     vertices;
    std::vector<TriElement> triangles;

    Mesh() = default;
    ~Mesh();

    void readFromObj (std::string path);
    void setBuffers ();
    void draw (Shader &shader) const;
    void draw (Shader &shader, Transform &transform) const;

private:
    // OpenGL buffers
    unsigned int VAO, VBO, EBO;
};

std::ostream &operator<< (std::ostream &os, const Mesh &m);

#endif