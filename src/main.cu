#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using std::vector, std::string;

#include "Vec3.cuh"
#include "ParticleContainer.cuh"

int main() {

    using namespace std::string_literals;

    ParticleContainer pc("Carbon"s, 12.01, 0);

    int numParticles = 10;
    vector<float> x(numParticles, 0.0f);
    vector<float> y(numParticles, 0.0f);
    vector<float> z(numParticles, 0.0f);
    vector<float> ux(numParticles, 1.0f);
    vector<float> uy(numParticles, 0.5f);
    vector<float> uz(numParticles, 0.25f);
    vector<float> w(numParticles, 1.0f);

    pc.addParticles(x, y, z, ux, uy, uz, w);

    int numSteps = 1;
    float dt = 1.0f / numSteps;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numSteps; i++) {
        pc.push(dt);
    }

    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

    std::cout << "Average push duration: " << (float) duration / numSteps / 1000 << " us\n";

    for (int i = 0; i < pc.numParticles; i++) {
        pc.velocity_x[i] = 0.0;
    }

    std::cout << pc;

    pc.copyToCPU();
}
