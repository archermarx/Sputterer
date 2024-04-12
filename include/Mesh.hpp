#ifndef _MESH_H
#define _MESH_H

#include <vector>
#include <string>
#include "Vec3.hpp"
#include "Shader.hpp"
using std::vector, std::string;

class Mesh {
    public:
        int numVertices;
        int numElements;
        vector<Point3<float>> vertices;
        vector<Vec3<unsigned int>> elements;
        bool enabled;

        Mesh() = default;
        Mesh(string path);
        ~Mesh();

        void draw(Shader &shader);
        void enable();
        void disable();
    private:
        unsigned int VAO, VBO, EBO;
};

std::ostream &operator<<(std::ostream &os, Mesh const &m);

#endif