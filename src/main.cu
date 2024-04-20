// C++ headers
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "App.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "Mesh.hpp"
#include "Shader.hpp"
#include "Surface.hpp"
#include "Window.hpp"

#include "Cuda.cuh"

#include "Triangle.cuh"

#include "ParticleContainer.cuh"

#include "gl_helpers.hpp"

using std::vector, std::string;

int main (int argc, char *argv[]) {
    // Handle command line arguments
    string filename("input.toml");
    if (argc > 1) {
        filename = argv[1];
    }

    bool display(false);
    if (argc > 2) {
        string _display(argv[2]);
        display = static_cast<bool>(stoi(_display));
    }

    Window window("Sputterer", App::SCR_WIDTH, App::SCR_HEIGHT);

    glfwSetFramebufferSizeCallback(window.window, App::framebufferSizeCallback);
    glfwSetCursorPosCallback(window.window, App::mouseCursorCallback);
    glfwSetScrollCallback(window.window, App::scrollCallback);

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    Input input(filename);
    input.read();

    App::camera.orientation = glm::normalize(glm::vec3(input.chamberRadius));
    App::camera.distance    = 2 * input.chamberRadius;
    App::camera.yaw         = -135;
    App::camera.pitch       = 30;
    App::camera.updateVectors();

    for (const auto &surface : input.surfaces) {
        std::cout << surface.name << "\n";
        std::cout << surface.mesh << "\n";
    }

    glEnable(GL_DEPTH_TEST);

    ParticleContainer pc{"noname", 1.0f, 1};
    pc.addParticles(input.particle_x, input.particle_y, input.particle_z, input.particle_vx, input.particle_vy,
                    input.particle_vz, input.particle_w);

    glm::vec3 particleColor{0.0f, 0.2f, 0.8f};
    glm::vec3 particleScale{0.1f};

    Mesh particleMesh{};
    particleMesh.readFromObj("o_sphere.obj");
    particleMesh.setBuffers();

    // construct triangles
    auto mesh = input.surfaces.at(0).mesh;

    std::vector<Triangle> h_triangles;
    for (const auto &[i1, i2, i3] : mesh.triangles) {
        auto scale     = input.surfaces.at(0).scale;
        auto translate = input.surfaces.at(0).translate;

        auto model = glm::translate(glm::mat4(1.0), translate);
        model      = glm::scale(model, scale);

        auto v1 = make_float3(model * glm::vec4(mesh.vertices[i1].pos, 1.0));
        auto v2 = make_float3(model * glm::vec4(mesh.vertices[i2].pos, 1.0));
        auto v3 = make_float3(model * glm::vec4(mesh.vertices[i3].pos, 1.0));

        h_triangles.emplace_back(v1, v2, v3);
    }

    Ray ray{.origin    = {pc.position_x[0], pc.position_y[0], pc.position_z[0]},
            .direction = {pc.velocity_x[0] * input.timestep, pc.velocity_y[0] * input.timestep,
                          pc.velocity_z[0] * input.timestep}};

    std::cout << "Ray origin = " << ray.origin << ", direction = " << ray.direction << "\n\n";

    for (auto &t : h_triangles) {
        auto info = hits_triangle(ray, t);
        std::cout << "Triangle: " << t.v0 << ", " << t.v1 << ", " << t.v2 << "\n";
        std::cout << "Hit: " << (info.hits ? "true" : "false") << "\n";
        if (info.hits) {
            std::cout << "t = " << info.t << ", "
                      << "norm = " << info.norm << "\n"
                      << "intersect = " << ray.origin + info.t * ray.direction << "\n";
        }
        std::cout << "\n";
    }

    cuda::vector<Triangle> d_triangles{h_triangles};

    size_t id = 0;

    while (window.open && display) {
        // process user input
        float currentFrame = glfwGetTime();
        App::deltaTime     = currentFrame - App::lastFrame;
        App::lastFrame     = currentFrame;
        App::processInput(window.window);

        // if (id < 5) {
        //     std::cout << "Pos, vel : " << std::endl;
        //     std::cout << pc.position_x[0] << ", " << pc.position_y[0] << ", " << pc.position_z[0] << "\n";
        //     std::cout << pc.velocity_x[0] << ", " << pc.velocity_y[0] << ", " << pc.velocity_z[0] << "\n";
        // }

        // Push particles
        pc.push(input.timestep, d_triangles);

        // Copy back to CPU
        pc.copyToCPU();

        // draw background
        glClearColor(0.4f, 0.5f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // update camera projection
        shader.updateView(App::camera, App::aspectRatio);

        for (const auto &surface : input.surfaces) {
            // set the model matrix
            Transform t(surface.color, surface.scale, surface.translate);
            surface.mesh.draw(shader, t);
        }

        for (int i = 0; i < pc.numParticles; i++) {
            Transform t;
            t.scale     = particleScale;
            t.translate = glm::vec3{pc.position_x[i], pc.position_y[i], pc.position_z[i]};
            t.color     = particleColor;
            particleMesh.draw(shader, t);
        }

        window.checkForUpdates();
        id += 1;
    }

    return 0;
}
