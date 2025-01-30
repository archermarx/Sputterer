#pragma once
#ifndef SPUTTERER_PARTICLECONTAINER_H
#define SPUTTERER_PARTICLECONTAINER_H

// STL headers
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// CUDA headers
#include <curand.h>
#include <curand_kernel.h>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <glm/glm.hpp>
#include "Surface.h"
#include "Shader.h"
#include "cuda_helpers.h"
#include "Triangle.h"

using thrust::host_vector, thrust::device_vector;

using std::vector, std::string;

constexpr size_t MAX_PARTICLES = 35'000'000;

// forward decl
class Input;

struct DeviceParticleContainer {
    float3 *position;
    float3 *velocity;
    float *weight;
    int num_particles;
    curandState *rng;
};

constexpr float DEFAULT_SCALE = 0.1f;
const glm::vec3 DEFAULT_COLOR = {0.2f, 0.2f, 0.2f};

class ParticleContainer {
    // Holds information for many particles of a specific species.
    // Species are differentiated by charge state and mass.

  public:
    string name = "noname"; // name of particles
    double mass = 0.0;      // mass in atomic mass units
    int charge = 0;         // charge number
    int num_particles{0};   // number of particles in container

    // RNG state
    device_vector<curandState> d_rng;

    // Position in meters
    host_vector<float3> position;
    device_vector<float3> d_position;

    // Velocity in m/s
    host_vector<float3> velocity;
    device_vector<float3> d_velocity;

    // Particle weight (computational particles per real particle
    host_vector<float> weight;
    device_vector<float> d_weight;

    device_vector<float> d_tmp;

    // Particle mesh
    glm::vec3 color = DEFAULT_COLOR;
    float scale = DEFAULT_SCALE;
    Mesh mesh{};
    ShaderProgram shader;
    bool render = true;

    void draw (Camera &cam, float aspect_ratio);

    void setup_shaders (glm::vec3 color = DEFAULT_COLOR, float scale = DEFAULT_SCALE);

    // Constructor
    ParticleContainer () {};
    ParticleContainer (string name, size_t capacity = MAX_PARTICLES, double mass = 0.0, int charge = 0);

    // allocate memory and set up rng
    void initialize (size_t capacity);

    // push particles to next positions (for now just use forward Euler)
    void evolve (Scene scene, const device_vector<Material> &mats, const device_vector<size_t> &ids,
                 device_vector<int> &collected, const device_vector<HitInfo> &hits,
                 const device_vector<float> &num_emit, const Input &input);

    // add particles to the container
    void add_particles (const host_vector<float3> &pos, const host_vector<float3> &vel, const host_vector<float> &w);

    // Returns kernel launch params
    template <class T>
    std::pair<dim3, dim3> get_kernel_launch_params (size_t num_elems, T func) const {
        int min_grid_size, block_size;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func));
        auto grid_size = static_cast<int>(ceil(static_cast<float>(num_elems) / static_cast<float>(block_size)));
        grid_size = std::max(grid_size, min_grid_size);
        dim3 grid(grid_size, 1, 1);
        dim3 block(block_size, 1, 1);
        return std::make_pair(grid, block);
    }
    // Set particles that leave bounds to have negative weights and remove them
    void remove_out_of_bounds (const Input &input);

    // Copy particles on GPU to CPU
    void copy_to_cpu ();

  private:
    unsigned int buffer{};

    [[nodiscard]] DeviceParticleContainer data ();
};

float rand_uniform (float min = 0.0f, float max = 1.0f);

float rand_normal (float mean = 0.0f, float std = 1.0f);

__host__ __device__ float carbon_diffuse_prob (float cos_incident_angle, float incident_energy_ev);

__device__ float3 sample_diffuse (const Triangle &tri, float3 norm, float thermal_speed, curandState *rng);

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc);

#endif
