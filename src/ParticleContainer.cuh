
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
    string name;         // name of particles
    double mass;         // mass in atomic mass units
    int    charge;       // charge number
    int    numParticles; // number of particles in container

    // Position in meters
    vector<float> position_x;
    vector<float> position_y;
    vector<float> position_z;
    float        *d_pos_x, *d_pos_y, *d_pos_z;

    // Velocity in m/s
    vector<float> velocity_x;
    vector<float> velocity_y;
    vector<float> velocity_z;
    float        *d_vel_x, *d_vel_y, *d_vel_z;

    // Particle weight (computational particles per real particle
    vector<float> weight;
    float        *d_weight;

    // Constructor
    ParticleContainer(string name, double mass, int charge);

    // push particles to next positions (for now just use forward Euler)
    void push (const float dt, const cuda::vector<Triangle> &tris);

    // add particles to the container
    void addParticles (vector<float> x, vector<float> y, vector<float> z, vector<float> vx, vector<float> vy,
                       vector<float> vz, vector<float> w);

    // Copy particles on GPU to CPU
    void copyToCPU ();

    ~ParticleContainer();
};

std::ostream &operator<< (std::ostream &os, ParticleContainer const &pc);

#endif
