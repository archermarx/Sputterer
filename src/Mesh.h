#ifndef SPUTTERER_MESH_H
#define SPUTTERER_MESH_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using std::string, std::vector;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 norm;

    Vertex() = default;

    Vertex(const glm::vec3 &position, const glm::vec3 &normal) {
        pos = position;
        norm = normal;
    }
};

std::ostream &operator<< (std::ostream &os, const Vertex &v);

struct TriElement {
    unsigned int i1, i2, i3;

    TriElement() = default;

    TriElement(size_t a, size_t b, size_t c) {
        i1 = a, i2 = b, i3 = c;
    }
};

std::ostream &operator<< (std::ostream &os, const TriElement &t);

struct Transform {
    glm::vec3 scale{1.0};
    glm::vec3 translate{0.0, 0.0, 0.0};
    glm::vec3 rotation_axis{0.0, 1.0, 0.0};
    float rotation_angle{0.0};

    Transform() = default;

    [[maybe_unused]] Transform(glm::vec3 scale, glm::vec3 translate, glm::vec3 rotation_axis, float rotation_angle)
        : scale(scale)
        , translate(translate)
        , rotation_axis(glm::normalize(rotation_axis))
        , rotation_angle(rotation_angle) {}

    [[nodiscard]] glm::mat4 get_matrix () const {
        glm::mat4 model{1.0f};
        model = glm::translate(model, translate);
        model = glm::rotate(model, glm::radians(rotation_angle), rotation_axis);
        model = glm::scale(model, scale);
        return model;
    }
};

class Mesh {
  public:
    size_t num_vertices{0};
    size_t num_triangles{0};

    bool smooth{false};
    bool buffers_set{false};

    vector<Vertex> vertices{};
    vector<TriElement> triangles{};

    Mesh() = default;

    ~Mesh();

    void load (const string &path);
    void read_from_obj (const string &path);
    void read_from_str (const string &str);

    void set_buffers ();

    void draw () const;

    // Vertex array buffer
    // Public so we can access this from InstancedArray
    unsigned int vao{}, ebo{};

  private:
    // OpenGL buffers
    unsigned int vbo{};
};

std::ostream &operator<< (std::ostream &os, const Mesh &m);

#endif
