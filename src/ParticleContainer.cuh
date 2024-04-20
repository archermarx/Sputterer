
#ifndef _PARTICLE_CONTAINER_CUH
#define _PARTICLE_CONTAINER_CUH

#include <iostream>
#include <string>
#include <vector>

#include "Cuda.cuh"
#include "Triangle.cuh"

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
    vector<float3>       position;
    cuda::vector<float3> d_position{MAX_PARTICLES};

    // Velocity in m/s
    vector<float3>       velocity;
    cuda::vector<float3> d_velocity{MAX_PARTICLES};

    // Particle weight (computational particles per real particle
    vector<float>       weight;
    cuda::vector<float> d_weight{MAX_PARTICLES};

    // Constructor
    ParticleContainer(string name, double mass, int charge);

    // push particles to next positions (for now just use forward Euler)
    void push (const float dt, const cuda::vector<Triangle> &tris);

    // add particles to the container
    void addParticles (vector<float> x, vector<float> y, vector<float> z, vector<float> vx, vector<float> vy,
                       vector<float> vz, vector<float> w);

    // Copy particles on GPU to CPU
    void copyToCPU ();

    ~ParticleContainer() = default;
};

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc);

#endif
