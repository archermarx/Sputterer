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
#include "ParticleContainer.cuh"
#include "Triangle.cuh"

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

    // TRIANGLE TESTS

    std::vector<Triangle> host_tris{
        {.p1 = {0.0, 0.0, 0.0}, .p2 = {0.0, 1.0, 0.0}, .p3 = {1.0, 0.0, 0.0}},
        {.p1 = {1.0, 1.0, 0.0}, .p2 = {1.0, 1.0, 1.0}, .p3 = {1.0, 0.0, 1.0}},
        {.p1 = {1.0, 0.0, 1.0}, .p2 = {1.0, 0.5, 0.1}, .p3 = {2.0, 5.0, 3.0}},
    };

    cuda::vector<Triangle> dev_tris(host_tris);

    std::cout << dev_tris.size() << std::endl;
    // invoke our kernel
    g_translate_triangles<<<1, 3>>>(dev_tris.data(), dev_tris.size());

    // Copy results back to host
    dev_tris.copyTo(host_tris);

    // let's see what we have
    for (size_t i = 0; i < host_tris.size(); i++) {
        std::cout << "Triangle: " << i << "\n";
        std::cout << "[" << host_tris[i].p1.x << ", " << host_tris[i].p1.y << ", " << host_tris[i].p1.z << "]\n";
        std::cout << "[" << host_tris[i].p2.x << ", " << host_tris[i].p2.y << ", " << host_tris[i].p2.z << "]\n";
        std::cout << "[" << host_tris[i].p3.x << ", " << host_tris[i].p3.y << ", " << host_tris[i].p3.z << "]\n\n";
    }

    while (window.open && display) {
        // process user input
        float currentFrame = glfwGetTime();
        App::deltaTime     = currentFrame - App::lastFrame;
        App::lastFrame     = currentFrame;
        App::processInput(window.window);

        // Push particles
        pc.push(input.timestep);

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
    }

    return 0;
}
