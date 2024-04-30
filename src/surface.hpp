#pragma once
#ifndef SPUTTERER_SURFACE_HPP
#define SPUTTERER_SURFACE_HPP

#include <string>
#include <vector>

#include "mesh.hpp"
#include "vec3.hpp"

using std::vector, std::string;

struct material {
  bool collect{false};
  float sticking_coeff{0.0f};
  float diffuse_coeff{0.0f};
  float temperature_k{300.0f};
};

struct emitter {
  bool emit{false};
  float flux{0.0};
  float velocity{1.0};
  float spread{0.1};
  bool reverse{false};
};

struct surface {
  // Name of surface
  string name{"noname"};

  // Emitter options
  emitter emitter{};

  // Material options
  material material{};

  // Geometric options
  mesh mesh{};
  transform transform{};
  vec3 color{0.5f, 0.5f, 0.5f};
};

#endif