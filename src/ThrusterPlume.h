#ifndef SPUTTERER_THRUSTERPLUME_H
#define SPUTTERER_THRUSTERPLUME_H

#include <array>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <glm/glm.hpp>

#include "PlumeInputs.h"
#include "Shader.h"
#include "Triangle.h"
#include "ParticleContainer.h"

class Input;

struct CurrentFraction {
    double main;
    double scattered;
    double cex;
};

class ThrusterPlume {
  public:
    PlumeInputs inputs;

    ParticleContainer particles{};

    // Shaders
    ShaderProgram cone_shader;
    ShaderProgram particle_shader;

    // display options
    bool render = true;

    void find_hits (Input &input, Scene &h_scene, host_vector<Material> &h_materials,
                    host_vector<size_t> &h_material_ids, host_vector<HitInfo> &hits, host_vector<float3> &hit_positions,
                    host_vector<float> &num_emit);

    [[nodiscard]] double main_divergence_angle () const;

    [[nodiscard]] double scattered_divergence_angle () const;

    // convert 3D Cartesian coordinates (x, y, z) to thruster-relative polar coordinates (r, alpha)
    [[nodiscard]] [[maybe_unused]] glm::vec2 convert_to_thruster_coords (glm::vec3 position) const;

    [[nodiscard]] CurrentFraction current_fractions () const;

    ThrusterPlume(PlumeInputs inputs);
    void setup_shaders (float len);
    void draw (Camera &cam, float aspect_ratio);
    CurrentFraction current_density (glm::vec2 coords) const;
    void probe() const;

  private:
    unsigned int vbo{}, vao{};
};

struct Species;

double sputtering_yield (double energy, double angle, Species incident, Species target);

#endif // SPUTTERER_THRUSTERPLUME_H
