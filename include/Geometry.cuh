#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#include <vector>
#include <string>
#include "Vec3.cuh"
using std::vector, std::string;

class Geometry {
    public:
        string name;
        bool emits;
        bool collects;
        float temperature;
        vector<Point3<float>> vertices;
        vector<Vec3<int>>   elements;

        Geometry::Geometry(string name, string path, bool emits, bool collects, float temperature);
        Geometry::Geometry(string name, string path, bool emits, bool collects);


};

#endif