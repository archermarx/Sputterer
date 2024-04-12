#ifndef _MESH_H
#define _MESH_H

#include <vector>
#include <string>
#include "Vec3.hpp"
using std::vector, std::string;

class Mesh {
    public:
        int numVertices;
        int numElements;
        vector<Point3<float>> vertices;
        vector<Vec3<int>>   elements;
        Mesh(string path);
};

std::ostream &operator<<(std::ostream &os, Mesh const &m);

#endif