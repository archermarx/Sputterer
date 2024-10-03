#ifndef SPUTTERER_SURFACE_H
#define SPUTTERER_SURFACE_H

#include <string>
#include <vector>

#include "Mesh.h"
#include "Shader.h"
#include "vec3.h"

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

struct SceneGeometry {
    vector<Surface> surfaces;    
    Shader shader;
    void setup_shaders();
    void draw(Camera &camera, float aspect_ratio);
};

#endif
