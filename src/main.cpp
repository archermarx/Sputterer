#include <iostream>
#include <vector>
#include <string>

using std::vector, std::string;

#include "ParticleContainer.hpp"

int main() {

    using namespace std::string_literals;

    ParticleContainer<float> pc("Carbon"s, 12.01f, 0);

    vector<float> x = {0.0f};
    vector<float> y = {0.0f};
    vector<float> z = {0.0f};
    vector<float> ux = {1.0f};
    vector<float> uy = {0.5f};
    vector<float> uz = {0.25f};
    vector<float> w = {1.0f};

    pc.addParticles(x, y, z, ux, uy, uz, w);
    std::cout << pc;

    pc.push(1.0f);
    std::cout << pc;
}
