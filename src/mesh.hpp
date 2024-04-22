#ifndef _MESH_HPP
#define _MESH_HPP

// Standard libraries
#include <cstddef>
#include <string>
#include <vector>

#include "shader.hpp"
#include "vec3.hpp"

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
    vec3  scale{1.0};
    vec3  translate{0.0, 0.0, 0.0};
    vec3  rotationAxis{0.0, 0.0, 1.0};
    float rotationAngle{0.0};

    Transform() = default;

    Transform(vec3 scale, vec3 translate, vec3 rotationAxis, float rotationAngle)
        : scale(scale)
        , translate(translate)
        , rotationAxis(glm::normalize(rotationAxis))
        , rotationAngle(rotationAngle) {}

    glm::mat4 getMatrix () const {
        glm::mat4 model{1.0f};
        model = glm::translate(model, translate);
        model = glm::rotate(model, glm::radians(rotationAngle), rotationAxis);
        model = glm::scale(model, scale);
        return model;
    }
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
    void draw (const Shader &shader, const Transform &transform, const vec3 &color) const;

private:
    // OpenGL buffers
    unsigned int VAO, VBO, EBO;
};

std::ostream &operator<< (std::ostream &os, const Mesh &m);

#endif