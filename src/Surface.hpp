#pragma once
#ifndef SPUTTERER_SURFACE_HPP
#define SPUTTERER_SURFACE_HPP

#include <string>
#include <vector>

#include "Mesh.hpp"
#include "vec3.hpp"

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
  vec3 color{0.5f, 0.5f, 0.5f};
};

#endif
