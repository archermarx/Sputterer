#include "Surface.h"
#include "Camera.h"
#include "Shader.h"
#include "ShaderCode.h"

void SceneGeometry::setup_shaders() {
    shader.load(shaders::mesh.vert, shaders::mesh.frag);
    for (auto &surf: surfaces) {
        surf.mesh.set_buffers();
    }
}
void SceneGeometry::draw(Camera &camera, float aspect_ratio) {
    shader.use();
    shader.update_view(camera, aspect_ratio);
    for (const auto &surface: surfaces) {
        shader.set_uniform("model", surface.transform.get_matrix());
        shader.set_uniform("objectColor", surface.color);
        surface.mesh.draw();
    }
}
