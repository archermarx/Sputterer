#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using std::vector, std::string;

#include "ParticleContainer.hpp"

int main() {

    using namespace std::string_literals;

    ParticleContainer<float> pc("Carbon"s, 12.01f, 0);

    int numParticles = 1000000;
    vector<float> x(numParticles, 0.0f);
    vector<float> y(numParticles, 0.0f);
    vector<float> z(numParticles, 0.0f);
    vector<float> ux(numParticles, 1.0f);
    vector<float> uy(numParticles, 0.5f);
    vector<float> uz(numParticles, 0.25f);
    vector<float> w(numParticles, 1.0f);

    pc.addParticles(x, y, z, ux, uy, uz, w);

    int numSteps = 1000;
    float dt = 1.0f / numSteps;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numSteps; i++) {
        pc.push(dt, 4);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    std::cout << duration << std::endl;
    std::cout << "Average push duration: " << (float) duration / numSteps << " ms\n";
}
