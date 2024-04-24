//
// Created by marksta on 4/24/24.
//

#include <iostream>
#include "instanced_mesh.hpp"
#include "gl_helpers.hpp"

void InstancedMesh::setBuffers() {
    // enable buffer
    this->mesh.setBuffers();
    glGenBuffers(1, &this->buffer);
}

void InstancedMesh::draw(Shader shader) {

    // Bind vertex array
    auto VAO = this->mesh.VAO;
    GL_CHECK(glBindVertexArray(VAO));

    // Send over model matrix data
    auto matVectorSize = static_cast<GLsizei>(this->numInstances * sizeof(vec3));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, this->buffer));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, matVectorSize, &positions[0], GL_DYNAMIC_DRAW));

    // Set attribute pointers for translation
    GL_CHECK(glEnableVertexAttribArray(2));
    GL_CHECK(glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), nullptr));
    GL_CHECK(glVertexAttribDivisor(2, 1));

    // Bind element array buffer
    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->mesh.EBO));

    // Draw meshes
    shader.use();
    GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(3 * this->mesh.numTriangles),
                                     GL_UNSIGNED_INT, nullptr, numInstances));

    // unbind buffers
    GL_CHECK(glBindVertexArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}
