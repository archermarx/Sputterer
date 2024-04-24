#pragma once
#ifndef SPUTTERER_INSTANCED_MESH_HPP
#define SPUTTERER_INSTANCED_MESH_HPP

#include <vector>

#include "vec3.hpp"
#include "mesh.hpp"

using std::vector;

// TODO: just move this into particle container and have a particle_container.draw() method
// this would avoid copying the data twice
class InstancedMesh {
public:
    GLsizei           numInstances{0};
    Mesh              mesh{};
    vector<glm::vec3> positions;

    explicit InstancedMesh(size_t n)
        : positions(n, glm::vec3{0.0}) {}

    void draw (Shader shader);
    void setBuffers ();

private:
    unsigned int buffer{};
};

#endif // SPUTTERER_INSTANCED_MESH_HPP
