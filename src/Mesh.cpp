
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <glad/glad.h>

#include "Vec3.hpp"
#include "Mesh.hpp"
#include "gl_helpers.hpp"

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
        float x, y, z;
        unsigned int e1, e2, e3;

        if (firstChar == 'v') {
            // Read vertex points from file
            iss >> v >> x >> y >> z;
            vertices.emplace_back(x, y, z);
            numVertices += 1;
        } else if (firstChar == 'f') {
            // Read elements from file.
            // Subtract one from each index (obj files start at 1, opengl at 0)
            iss >> v >> e1 >> e2 >> e3;
            elements.emplace_back(e1 - 1, e2 - 1, e3 - 1);
            numElements += 1;
        }
    }
}

void Mesh::enable() {

    auto vertSize = numVertices * sizeof(Vec3<float>);
    auto elemSize = numElements * sizeof(Vec3<unsigned int>);

    // Set up buffers
    GL_CHECK( glGenVertexArrays(1, &VAO) );
    GL_CHECK( glGenBuffers(1, &VBO) );
    GL_CHECK( glGenBuffers(1, &EBO) );

    // Assign vertex data
    GL_CHECK( glBindVertexArray(VAO) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, VBO) );
    GL_CHECK( glBufferData(GL_ARRAY_BUFFER, vertSize, vertices.data(), GL_STATIC_DRAW) );

    // Assign element data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elemSize, elements.data(), GL_STATIC_DRAW);

    // Vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3<float>), (void*) 0);

    glBindVertexArray(0);

    enabled = true;
}

void Mesh::disable() {
    std::cout << "Disabling mesh" << std::endl;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    enabled = false;
}

void Mesh::draw(Shader &shader) {
    shader.use();
    // draw mesh
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, sizeof(Vec3<unsigned int>), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

Mesh::~Mesh() {
    if (enabled) {
        disable();
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
