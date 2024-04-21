// C++ headers
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Thrust headers
#include <thrust/host_vector.h>

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

    Window window("Sputterer", app::SCR_WIDTH, app::SCR_HEIGHT);

    glfwSetFramebufferSizeCallback(window.window, app::framebufferSizeCallback);
    glfwSetCursorPosCallback(window.window, app::mouseCursorCallback);
    glfwSetScrollCallback(window.window, app::scrollCallback);

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    Input input(filename);
    input.read();

    app::camera.orientation = glm::normalize(glm::vec3(input.chamberRadius));
    app::camera.distance    = 2 * input.chamberRadius;
    app::camera.yaw         = -135;
    app::camera.pitch       = 30;
    app::camera.updateVectors();

    // for (const auto &surface : input.surfaces) {
    //     std::cout << surface.name << "\n";
    //     std::cout << surface.mesh << "\n";
    // }

    glEnable(GL_DEPTH_TEST);

    // Create particle container, including any explicitly-specified initial particles
    ParticleContainer pc{"noname", 1.0f, 1};
    pc.addParticles(input.particle_x, input.particle_y, input.particle_z, input.particle_vx, input.particle_vy,
                    input.particle_vz, input.particle_w);

    glm::vec3 particleColor{0.2f, 0.2f, 0.2f};
    glm::vec3 particleColorOOB{1.0f, 0.2f, 0.2f};
    glm::vec3 particleScale{0.05f};

    // Read mesh from file
    Mesh particleMesh{};
    particleMesh.readFromObj("o_sphere.obj");
    particleMesh.setBuffers();

    // construct triangles
    std::vector<Triangle> h_triangles;
    for (const auto &surf : input.surfaces) {
        const auto &mesh      = surf.mesh;
        const auto &translate = surf.translate;
        const auto &scale     = surf.scale;
        for (const auto &[i1, i2, i3] : mesh.triangles) {
            auto model = glm::translate(glm::mat4(1.0), translate);
            model      = glm::scale(model, scale);

            auto v1 = make_float3(model * glm::vec4(mesh.vertices[i1].pos, 1.0));
            auto v2 = make_float3(model * glm::vec4(mesh.vertices[i2].pos, 1.0));
            auto v3 = make_float3(model * glm::vec4(mesh.vertices[i3].pos, 1.0));

            h_triangles.emplace_back(v1, v2, v3);
        }
    }

    thrust::device_vector<Triangle> d_triangles{h_triangles};

    // Create timing objects
    size_t      frame = 0, timingInterval = 100;
    float       totalTimeCompute = 0.0f, totalTime = 0.0f;
    cuda::event start{}, stopCompute{}, stopCopy{};

    while (true && window.open && display) {
        // process user input
        float currentFrame = glfwGetTime();
        app::deltaTime     = currentFrame - app::lastFrame;
        app::lastFrame     = currentFrame;
        app::processInput(window.window);

        auto physicalTimestep = input.timestep * app::deltaTime;

        // record compute start time
        if (frame > 1) {
            start.record();

            // Emit particles
            size_t triCount;
            for (const auto &surf : input.surfaces) {
                if (!surf.emit) {
                    continue;
                }

                for (size_t i = 0; i < surf.mesh.numTriangles; i++) {
                    pc.emit(h_triangles.at(i), surf.emitter_flux, physicalTimestep);
                }
                triCount += surf.mesh.numTriangles;
            }

            // Push particles
            pc.push(physicalTimestep, d_triangles);

            // Remove particles that are out of bounds
            pc.flagOutOfBounds(input.chamberRadius, input.chamberLength);

            stopCompute.record();

            // Copy back to CPU
            pc.copyToCPU();

            stopCopy.record();

            float elapsedCompute, elapsedCopy;
            elapsedCompute = cuda::eventElapsedTime(start, stopCompute);
            elapsedCopy    = cuda::eventElapsedTime(start, stopCopy);

            totalTime += elapsedCopy;
            totalTimeCompute += elapsedCompute;
            float computePercentage = totalTimeCompute / totalTime * 100;

            if (frame % timingInterval == 0 && frame > 0) {
                std::cout << "Average compute time: " << totalTime / frame << "ms (" << computePercentage
                          << "% compute)\n";
                std::cout << "Number of particles: " << pc.numParticles << std::endl;
            }
        }

        // draw background
        glClearColor(0.4f, 0.5f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // update camera projection
        shader.updateView(app::camera, app::aspectRatio);

        for (const auto &surface : input.surfaces) {
            // set the model matrix
            Transform t(surface.color, surface.scale, surface.translate);
            surface.mesh.draw(shader, t);
        }

        for (int i = 0; i < pc.numParticles; i++) {
            Transform t;
            t.scale     = particleScale;
            t.translate = glm::vec3{pc.position[i].x, pc.position[i].y, pc.position[i].z};
            t.color     = pc.weight[i] > 0 ? particleColor : particleColorOOB;
            particleMesh.draw(shader, t);
        }

        window.checkForUpdates();
        frame += 1;
    }

    std::cout << pc << std::endl;

    return 0;
}
