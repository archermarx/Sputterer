#ifndef _MESH_HPP
#define _MESH_HPP

// Standard libraries
#include <cstddef>
#include <string>
#include <vector>

// GLM types
#include "Shader.hpp"
#include "Vec3.hpp"

struct Vertex {
    vec3 pos;
    vec3 norm;
};

std::ostream &operator<< (std::ostream &os, const Vertex &v);

struct TriElement {
    unsigned int i1, i2, i3;
};

std::ostream &operator<< (std::ostream &os, const TriElement &t);

struct Transform {
    vec3 color{0.3, 0.3, 0.3};
    vec3 scale{1.0};
    vec3 translate{0.0, 0.0, 0.0};
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