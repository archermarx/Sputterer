#pragma once

#ifndef SPUTTERER_MESH_HPP
#define SPUTTERER_MESH_HPP

// Standard libraries
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

#include "shader.hpp"
#include "vec3.hpp"

using std::string, std::vector;

struct vertex {
  vec3 pos;
  vec3 norm;
};

std::ostream &operator<< (std::ostream &os, const vertex &v);

struct tri_element {
  unsigned int i1, i2, i3;
};

std::ostream &operator<< (std::ostream &os, const tri_element &t);

struct transform {
  vec3 scale{1.0};
  vec3 translate{0.0, 0.0, 0.0};
  vec3 rotation_axis{0.0, 1.0, 0.0};
  float rotation_angle{0.0};

  transform () = default;

  [[maybe_unused]] transform (vec3 scale, vec3 translate, vec3 rotation_axis, float rotation_angle)
          : scale(scale), translate(translate), rotation_axis(glm::normalize(rotation_axis)),
            rotation_angle(rotation_angle) {}

  [[nodiscard]] glm::mat4 get_matrix () const {
    glm::mat4 model{1.0f};
    model = glm::translate(model, translate);
    model = glm::rotate(model, glm::radians(rotation_angle), rotation_axis);
    model = glm::scale(model, scale);
    return model;
  }
};

class mesh {
public:
  size_t num_vertices{0};
  size_t num_triangles{0};

  bool smooth{false};
  bool buffers_set{false};

  vector<vertex> vertices{};
  vector<tri_element> triangles{};

  mesh () = default;

  ~mesh ();

  void read_from_obj (const string &path);

  void set_buffers ();

  void draw (shader &shader) const;

  void draw (const shader &shader, const transform &transform, const vec3 &color) const;

  // Vertex array buffer
  // Public so we can access this from InstancedArray
  unsigned int vao{}, ebo{};

private:
  // OpenGL buffers
  unsigned int vbo{};
};

std::ostream &operator<< (std::ostream &os, const mesh &m);

#endif