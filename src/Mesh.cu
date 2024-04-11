
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

#include "Vec3.cuh"
#include "Mesh.cuh"

using std::string, std::vector;

Mesh::Mesh(string path) : numVertices(0), numElements(0) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "File " << path << " does not exist!\n";
        throw;
    }

    std::ifstream objFile(path);

    char firstChar;
    string line, v;

    while (!objFile.eof()) {

        // Read a line from the file
        std::getline(objFile, line);
        std::istringstream iss(line);

        firstChar = iss.peek();
        if (firstChar == 'v') {
            Vec3<float> vert{};
            iss >> v >> vert.x >> vert.y >> vert.z;
            vertices.push_back(vert);
            numVertices += 1;
        } else if (firstChar == 'f') {
            Vec3<int> elem{};
            iss >> v >> elem.x >> elem.y >> elem.z;
            elements.push_back(elem);
            numElements += 1;
        }
    }
}

std::ostream& operator <<(std::ostream &os, Mesh const &m) {
    os << "Vertices\n=======================\n";
    for (int i = 0; i < m.numVertices; i++) {
        os << i << ": " << m.vertices[i] << "\n";
    }

    os << "Elements\n=======================\n";
    for (int i = 0; i < m.numElements; i++) {
        std::cout << i << ": " << m.elements[i] << "\n";
    }

    return os;
}
