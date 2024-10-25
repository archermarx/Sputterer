#ifndef SPUTTERER_SURFACE_H
#define SPUTTERER_SURFACE_H

#include <string>
#include <vector>
#include <glm/vec3.hpp>

#include "Mesh.h"
#include "Shader.h"

using std::vector, std::string;

struct Material {
  bool collect{false};
  bool sputter{true};
  float sticking_coeff{0.0f};
  float diffuse_coeff{0.0f};
  float temperature_K{300.0f};
};

struct Surface {
  // Name of surface
  string name{"noname"};

  // Material options
  Material material{};

  // Geometric options
  Mesh mesh{};
  Transform transform{};
  glm::vec3 color{0.5f, 0.5f, 0.5f};
  bool has_current_density{false};
  glm::vec3 current_density{0.0f, 0.0f, 0.0f};
};

#endif
