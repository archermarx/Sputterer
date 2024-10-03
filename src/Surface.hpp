#pragma once
#ifndef SPUTTERER_SURFACE_HPP
#define SPUTTERER_SURFACE_HPP

#include <string>
#include <vector>

#include "Camera.hpp"
#include "Mesh.hpp"
#include "Shader.hpp"
#include "ShaderCode.hpp"
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

struct SceneGeometry {
    vector<Surface> surfaces;    
    Shader shader;
    void setup_shaders() {
        shader.load(shaders::mesh.vert, shaders::mesh.frag);
        for (auto &surf: surfaces) {
            surf.mesh.set_buffers();
        }
    }
    void draw(Camera camera, float aspect_ratio) {
        shader.use();
        shader.update_view(camera, aspect_ratio);
        for (const auto &surface: surfaces) {
            shader.set_uniform("model", surface.transform.get_matrix());
            shader.set_uniform("objectColor", surface.color);
            surface.mesh.draw();
        }
    }
};

#endif
