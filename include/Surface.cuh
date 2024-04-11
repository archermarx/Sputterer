#ifndef _SURFACE_H
#define _SURFACE_H

#include <string>
#include "Mesh.cuh"
using std::string;

class Surface {
    public:
        string name;
        bool emit;
        bool collect;
        Mesh mesh;
        Surface(string name, string path, bool emit, bool collect): name(name), mesh(path), emit(emit), collect(collect) {};
};

#endif