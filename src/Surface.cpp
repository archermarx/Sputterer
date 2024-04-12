
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
    std::vector<Vec3<float>> rawVertices;
    std::vector<Vec3<float>> rawElements;

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
            // put vertex in array
            rawVertices.emplace_back(x, y, z);
        } else if (firstChar == 'f') {
            // Read elements from file.
            // Subtract one from each index (obj files start at 1, opengl at 0)
            iss >> v >> e1 >> e2 >> e3;
            rawElements.emplace_back(e1 - 1, e2 - 1, e3 - 1);
        }
    }

    // now, split vertices for each face and generate normals
    int id = 0;
    for (const auto &[e1, e2, e3] : rawElements) {

        // Get vertex coords
        auto a = rawVertices.at(e1);
        auto b = rawVertices.at(e2);
        auto c = rawVertices.at(e3);

        // Compute normal
        auto n = (b - a).cross(c - a);
        n.normalize();

        // Add vertices to array
        vertices.emplace_back(a, n);
        vertices.emplace_back(b, n);
        vertices.emplace_back(c, n);

        // Add elements to array
        elements.emplace_back(id, id + 1, id + 2);

        // increment vertex and element numbers
        numVertices += 3;
        numElements += 1;
        id += 3;
    }
}

void Surface::enable() {

    unsigned int vertSize = numVertices * sizeof(Vertex);
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
    GL_CHECK( glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0) );

    // Vertex normals
    GL_CHECK( glEnableVertexAttribArray(1) );
    GL_CHECK( glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))));

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

    GL_CHECK( glDrawElements(GL_TRIANGLES, 3 * numElements, GL_UNSIGNED_INT, 0) );

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
        os << i << ", pos =  " << m.vertices[i].pos << "\n";
        os << i << ", normal =  " << m.vertices[i].normal << "\n";
    }

    os << "Elements\n=======================\n";
    for (int i = 0; i < m.numElements; i++) {
        std::cout << i << ": " << m.elements[i] << "\n";
    }

    return os;
}
