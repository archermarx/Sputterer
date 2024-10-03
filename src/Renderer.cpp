#include "Renderer.hpp"
#include "Shader.hpp"
#include "Camera.hpp"
#include "vec3.hpp"
#include "Triangle.cuh"

void BVHRenderer::setup_shaders () {
    draw_depth = scene->bvh_depth;
    shader.load(shaders::bvh.vert, shaders::bvh.frag, shaders::bvh.geom);
    shader.use();

    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    float points[] = {0.0, 0.0, 0.0};
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(points), 0);
    glBindVertexArray(0);
}

void BVHRenderer::draw_box (Shader &shader, BBox &box, unsigned int &vao, unsigned int &vbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    auto center = box.center();
    auto extent = box.ub - box.lb;
    float points[] = {center.x, center.y, center.z};
    shader.set_vec3("extent", {extent.x, extent.y, extent.z});
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, 1);
}

void BVHRenderer::draw_bvh (int depth, int node_idx) {
    if (depth == 0) {
        return;
    }

    auto &node = this->scene->nodes[node_idx];
    // draw current box
    draw_box(shader, node.box, this->vao, this->vbo);

    if (node.is_leaf()) {
        return;
    } else {
        // recursively draw children
        draw_bvh(depth - 1, node.left_first);
        draw_bvh(depth - 1, node.left_first + 1);
    }
}

void BVHRenderer::draw (Camera camera, float aspect_ratio) {
    if (draw_depth == 0 || !render) {
        return;
    }

    shader.use();
    shader.set_mat4("camera", camera.get_matrix(aspect_ratio));
    draw_bvh(draw_depth, 0);
}
