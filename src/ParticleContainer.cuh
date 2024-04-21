
#ifndef _PARTICLE_CONTAINER_CUH
#define _PARTICLE_CONTAINER_CUH

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Cuda.cuh"
#include "Triangle.cuh"

using thrust::host_vector, thrust::device_vector;

using std::vector, std::string;

#define MAX_PARTICLES 35'000'000

class ParticleContainer {
    // Holds information for many particles of a specific species.
    // Species are differentiated by charge state and mass.

public:
    string name;            // name of particles
    double mass;            // mass in atomic mass units
    int    charge;          // charge number
    int    numParticles{0}; // number of particles in container

    // Position in meters
    host_vector<float3>   position;
    device_vector<float3> d_position{MAX_PARTICLES};

    // Velocity in m/s
    host_vector<float3>   velocity;
    device_vector<float3> d_velocity{MAX_PARTICLES};

    // Particle weight (computational particles per real particle
    host_vector<float>   weight;
    device_vector<float> d_weight{MAX_PARTICLES};

    // Constructor
    ParticleContainer(string name, double mass, int charge);

    // push particles to next positions (for now just use forward Euler)
    void push (const float dt, const thrust::device_vector<Triangle> &tris);

    // add particles to the container
    void addParticles (vector<float> x, vector<float> y, vector<float> z, vector<float> vx, vector<float> vy,
                       vector<float> vz, vector<float> w);

    // Emit particles from a given triangle
    void emit (Triangle &triangle, float flux, float dt);

    // Returns kernel launch params
    std::pair<dim3, dim3> getKernelLaunchParams (size_t block_size = 32) const;

    // Set particles that leave bounds to have negative weights
    void flagOutOfBounds (float radius, float length);

    // Copy particles on GPU to CPU
    void copyToCPU ();

    ~ParticleContainer() = default;
};

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc);

#endif
