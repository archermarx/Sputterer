
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <glad/glad.h>

#include "Vec3.hpp"
#include "Surface.hpp"
#include "gl_helpers.hpp"

using std::string, std::vector;

Surface::Surface(string name, string path, bool emit, bool collect)
    : numVertices(0), numElements(0), name(name), emit(emit), collect(collect) {

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

void Surface::enable() {

    unsigned int vertSize = numVertices * sizeof(Vec3<float>);
    unsigned int elemSize = numElements * sizeof(Vec3<unsigned int>);
    GLint vertSize_actual = 0, elemSize_actual = 0;

    // Set up buffers
    GL_CHECK( glGenVertexArrays(1, &VAO) );
    GL_CHECK( glGenBuffers(1, &VBO) );
    GL_CHECK( glGenBuffers(1, &EBO) );

    // Assign vertex data
    GL_CHECK( glBindVertexArray(VAO) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, VBO) );
    GL_CHECK( glBufferData(GL_ARRAY_BUFFER, vertSize, vertices.data(), GL_STATIC_DRAW) );

    GL_CHECK( glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &vertSize_actual) );
    std::cout << "Vertex size: " << vertSize_actual << ", expected " << vertSize << std::endl;

    // Assign element data
    GL_CHECK( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO) );
    GL_CHECK( glBufferData(GL_ELEMENT_ARRAY_BUFFER, elemSize, elements.data(), GL_STATIC_DRAW) );

    GL_CHECK( glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &elemSize_actual) );
    std::cout << "Element size: " << elemSize_actual << ", expected " << elemSize << std::endl;

    // Vertex positions
    GL_CHECK( glEnableVertexAttribArray(0) );
    GL_CHECK( glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0) );

    // Unbind arrays and buffers
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    enabled = true;
}

void Surface::disable() {
    std::cout << "Disabling mesh" << std::endl;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    enabled = false;
}

void Surface::draw(Shader &shader) {
    shader.use();
    // draw mesh
    GL_CHECK( glBindVertexArray(VAO) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, VBO) );
    GL_CHECK( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO) );

    GL_CHECK( glDrawElements(GL_TRIANGLES, sizeof(Vec3<unsigned int>), GL_UNSIGNED_INT, 0) );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

Surface::~Surface() {
    if (enabled) {
        disable();
    }
}

std::ostream& operator <<(std::ostream &os, Surface const &m) {
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
