
#ifndef _PARTICLE_CONTAINER_H
#define PARTICLE_CONTAINER_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

using std::string, std::vector;

template <typename T>
class ParticleContainer {
    // Holds information for many particles of a specific species.
    // Species are differentiated by charge state and mass.

    public:
        string name;        // name of particles
        T mass;             // mass in atomic mass units
        int charge;         // charge number
        int numParticles;   // number of particles in container

        // Position in meters
        vector<T> position_x;
        vector<T> position_y;
        vector<T> position_z;

        // Velocity in m/s
        vector<T> velocity_x;
        vector<T> velocity_y;
        vector<T> velocity_z;

        // Particle weight (computational particles per real particle
        vector<T> weight;

        // Constructor
        ParticleContainer(string name, T mass, int charge): name(name), mass(mass), charge(charge), numParticles(0) {};

        // push particles to next positions (for now just use forward Euler)
        void push(T dt, int numThreads);

        void pushSome(T dt, int begin, int end);

        // add particles to the container
        void addParticles(vector<T> x, vector<T> y, vector<T> z, vector<T> vx, vector<T> vy, vector<T> vz, vector<T> w);

};

template <typename T>
void ParticleContainer<T>::addParticles(vector<T> x, vector<T> y, vector<T> z, vector<T> ux, vector<T> uy, vector<T> uz, vector<T> w) {

    auto N = std::min({x.size(), y.size(), z.size(), ux.size(), uy.size(), uz.size(), w.size()});

    numParticles += N;

    position_x.insert(position_x.end(), x.begin(), x.begin() + N);
    position_y.insert(position_y.end(), y.begin(), y.begin() + N);
    position_z.insert(position_z.end(), z.begin(), z.begin() + N);

    velocity_x.insert(velocity_x.end(), ux.begin(), ux.begin() + N);
    velocity_y.insert(velocity_y.end(), uy.begin(), uy.begin() + N);
    velocity_z.insert(velocity_z.end(), uz.begin(), uz.begin() + N);

    weight.insert(weight.end(), w.begin(), w.begin() + N);
}


template <typename T>
void ParticleContainer<T>::pushSome(T dt, int begin, int end) {
    for (int i = begin; i < end; i++) {
        position_x[i] += velocity_x[i] * dt;
        position_y[i] += velocity_y[i] * dt;
        position_z[i] += velocity_z[i] * dt;
    }
}

template <typename T>
void ParticleContainer<T>::push(T dt, int numThreads) {
    std::thread threads[numThreads];

    int chunk = numParticles / numThreads;
    int remainder = numParticles - numThreads * chunk;

    int start = 0;
    for (int i = 0; i < numThreads; i++) {
        int begin = start;
        int count = chunk + static_cast<int>(i < remainder);
        int end = begin + count;
        start = end;
        threads[i] = std::thread(&ParticleContainer<T>::pushSome, this, dt, begin, end);
    }

    for (int i = 0; i < numThreads; i++) {
        threads[i].join();
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, ParticleContainer<T> const &pc) {
    os << "==========================================================\n";
    os << "Particle container \"" << pc.name << "\"\n";
    os << "==========================================================\n";
	os << "Mass: " << pc.mass << "\n";
    os << "Charge: " << pc.charge << "\n";
    os << "Number of particles: " << pc.numParticles << "\n";
    os << "----------------------------------------------------------\n";
    os << "\tx\ty\tz\tvx\tvy\tvz\tw\t\n";
    os << "----------------------------------------------------------\n";
    for (int i = 0; i < pc.numParticles; i++) {
        os << "\t" << pc.position_x[i] << "\t";
        os << pc.position_y[i] << "\t";
        os << pc.position_z[i] << "\t";
        os << pc.velocity_x[i] << "\t";
        os << pc.velocity_y[i] << "\t";
        os << pc.velocity_z[i] << "\t";
        os << pc.weight[i] << "\t";
        os << "\n";
    }
    os << "==========================================================\n";

    return os;
}

#endif
