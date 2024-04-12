#ifndef _SURFACE_H
#define _SURFACE_H

#include <vector>
#include <string>
#include "Vec3.hpp"
#include "Shader.hpp"
using std::vector, std::string;

class Surface {
    public:
        int numVertices;
        int numElements;
        vector<Point3<float>> vertices;
        vector<Vec3<unsigned int>> elements;
        bool enabled;

        string name;
        bool emit;
        bool collect;

        Surface() = default;
        Surface(string name, string path, bool emit, bool collect);
        ~Surface();

        void draw(Shader &shader);
        void enable();
        void disable();
    private:
        unsigned int VAO, VBO, EBO;
};

std::ostream &operator<<(std::ostream &os, Surface const &m);

#endif