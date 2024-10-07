#include "Renderer.h"
#include "Shader.h"
#include "ShaderCode.h"
#include "Camera.h"
#include "Triangle.h"
#include "gl_helpers.h"

Renderer::Renderer (Input &input, Scene *scene, ThrusterPlume &plume,
        ParticleContainer &particles, std::vector<Surface> surfaces)
    : bvh(scene)
    , plume(plume)
    , particles(particles)
    , grid()
    , geometry(surfaces) {
    if (input.display) {
        particles.setup_shaders(carbon_particle_color, carbon_particle_scale);
        plume.setup_shaders(input.chamber_length_m / 2);
    }
}

void Renderer::draw (Input &input, Camera &camera, float aspect_ratio) {
    if (input.display) {
        geometry.draw(camera, aspect_ratio);
        particles.draw(camera, aspect_ratio);
        bvh.draw(camera, aspect_ratio);
        grid.draw(camera, aspect_ratio);
        plume.draw(camera, aspect_ratio);
    }
}

GeometryRenderer::GeometryRenderer(std::vector<Surface> &surfaces) : surfaces(surfaces) {
    shader.link(shaders::mesh, "geometry");
    for (auto &surf : surfaces) {
        surf.mesh.set_buffers();
    }
}

void GeometryRenderer::draw(Camera &camera, float aspect_ratio) {
    shader.use();
    shader.set_uniform("view", camera.get_view_matrix());
    shader.set_uniform("projection", camera.get_projection_matrix(aspect_ratio));
    shader.set_uniform("viewPos", camera.distance * camera.orientation, true);
    for (const auto &surface : surfaces) {
        shader.set_uniform("model", surface.transform.get_matrix());
        shader.set_uniform("objectColor", surface.color);
        surface.mesh.draw();
    }
}


GridRenderer::GridRenderer () {
    shader.link(shaders::grid, "grid");
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

void GridRenderer::draw_grid(Grid grid, int level, glm::vec3 center) {
    auto num_lines = static_cast<int>(2 * grid.scale / grid.spacing + 1) * 2;
    auto num_vertices = num_lines * 4;
    // subdividing to get around 256 vertex limit on geometry shaders
    if (num_vertices > 256) {
        auto new_scale = grid.scale / 2;
        Grid new_grid(grid);
        new_grid.scale = new_scale;
        draw_grid(new_grid, level, center + glm::vec3{ new_scale, 0.0,  new_scale});
        draw_grid(new_grid, level, center + glm::vec3{ new_scale, 0.0, -new_scale});
        draw_grid(new_grid, level, center + glm::vec3{-new_scale, 0.0, -new_scale});
        draw_grid(new_grid, level, center + glm::vec3{-new_scale, 0.0,  new_scale});
    } else {
        shader.set_uniform("grid_center", center);
        shader.set_uniform("grid_scale", grid.scale);
        shader.set_uniform("grid_spacing", grid.spacing);
        shader.set_uniform("linewidth", grid.linewidth);
        shader.set_uniform("linecolor", grid.color);
        shader.set_uniform("level", level);
        glDrawArrays(GL_POINTS, 0, 1);
    }
}

void GridRenderer::draw (Camera &camera, float aspect_ratio) {
    if (!enabled) return;
    shader.use();
    shader.set_uniform("camera", camera.get_matrix(aspect_ratio));
    shader.set_uniform("opacity", opacity);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    for (size_t i = 0; i < grids.size(); i++) {
        draw_grid(grids[i], i);
    }
}

BVHRenderer::BVHRenderer (Scene *scene) : draw_depth(scene->bvh_depth), scene(scene) {
    shader.link(shaders::bvh, "bvh");
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

void BVHRenderer::draw_box (ShaderProgram &shader, BBox &box, unsigned int &vao, unsigned int &vbo) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    auto center = box.center();
    auto extent = box.ub - box.lb;
    float points[] = {center.x, center.y, center.z};
    shader.set_uniform<glm::vec3>("extent", {extent.x, extent.y, extent.z});
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, 1);
}

void BVHRenderer::draw_bvh (int depth, int node_idx) {
    if (depth == 0) return;

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

void BVHRenderer::draw (Camera &camera, float aspect_ratio) {
    if (draw_depth == 0 || !enabled) return;

    shader.use();
    shader.set_uniform("camera", camera.get_matrix(aspect_ratio));
    draw_bvh(draw_depth, 0);
}
