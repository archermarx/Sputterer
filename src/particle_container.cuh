#pragma once
#ifndef PARTICLE_CONTAINER_CUH
#define PARTICLE_CONTAINER_CUH

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

#include "surface.hpp"

#include "cuda.cuh"
#include "triangle.cuh"

using thrust::host_vector, thrust::device_vector;

using std::vector, std::string;

constexpr size_t MAX_PARTICLES = 35'000'000;

class ParticleContainer {
    // Holds information for many particles of a specific species.
    // Species are differentiated by charge state and mass.

public:
    string name;            // name of particles
    double mass;            // mass in atomic mass units
    int    charge;          // charge number
    int    numParticles{0}; // number of particles in container

    // RNG state
    device_vector<curandState> d_rng;

    // Position in meters
    host_vector<float3>   position;
    device_vector<float3> d_position;

    // Velocity in m/s
    host_vector<float3>   velocity;
    device_vector<float3> d_velocity;

    // Particle weight (computational particles per real particle
    host_vector<float>   weight;
    device_vector<float> d_weight;

    device_vector<float> d_tmp;

    // Particle mesh
    Mesh mesh{};
    void draw (Shader shader);
    void setBuffers ();

    // Constructor
    ParticleContainer(string name, double mass, int charge);

    // push particles to next positions (for now just use forward Euler)
    void push (float dt, const thrust::device_vector<Triangle> &tris, const thrust::device_vector<size_t> &ids,
               const thrust::device_vector<Material> &mats, thrust::device_vector<int> &collected);

    // add particles to the container
    void addParticles (vector<float> x, vector<float> y, vector<float> z, vector<float> vx, vector<float> vy,
                       vector<float> vz, vector<float> w);

    // Emit particles from a given triangle
    void emit (Triangle &triangle, Emitter emitter, float dt);

    // Returns kernel launch params
    [[nodiscard]] std::pair<dim3, dim3> getKernelLaunchParams (size_t block_size = 32) const;

    // Set particles that leave bounds to have negative weights
    void flagOutOfBounds (float radius, float length);

    // Remove particles with negative weights
    void removeFlaggedParticles ();

    // Copy particles on GPU to CPU
    void copyToCPU ();

private:
    unsigned int buffer{};
};

__host__ __device__ float carbon_diffuse_prob (float cos_incident_angle, float incident_energy_eV);

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc);

#endif
