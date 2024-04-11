#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using std::vector, std::string;

#include "Vec3.cuh"
#include "Mesh.cuh"
#include "ParticleContainer.cuh"

int main(int argc, char *argv[]) {
    string filename(argv[1]);

    Mesh m(filename);
    std::cout << m;

    return 0;
}
