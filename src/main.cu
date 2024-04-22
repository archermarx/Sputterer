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

// Imgui headers
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

// My headers (c++)
#include "gl_helpers.hpp"
#include "App.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "Mesh.hpp"
#include "Shader.hpp"
#include "Surface.hpp"
#include "Window.hpp"

// My headers (CUDA)
#include "Cuda.cuh"
#include "ParticleContainer.cuh"
#include "Triangle.cuh"

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

    // ImGUI initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable keyboard controls

    // Setup platform/renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window.window, true);
    ImGui_ImplOpenGL3_Init();

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    Input input(filename);
    input.read();

    app::camera.orientation = glm::normalize(glm::vec3(input.chamberRadius));
    app::camera.distance    = 2 * input.chamberRadius;
    app::camera.yaw         = -135;
    app::camera.pitch       = 30;
    app::camera.updateVectors();

    glEnable(GL_DEPTH_TEST);

    // Create particle container, including any explicitly-specified initial particles
    ParticleContainer pc{"noname", 1.0f, 1};
    pc.addParticles(input.particle_x, input.particle_y, input.particle_z, input.particle_vx, input.particle_vy,
                    input.particle_vz, input.particle_w);

    glm::vec3 particleColor{0.1f, 0.1f, 0.1f};
    glm::vec3 particleColorOOB{1.0f, 0.2f, 0.2f};
    glm::vec3 particleScale{0.025f};

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
    size_t frame = 0;

    float avgTimeCompute = 0.0f, avgTimeTotal = 0.0f;
    float iterReset = 100;
    float timeConst = 1 / iterReset;

    cuda::event start{}, stopCompute{}, stopCopy{};

    while (true && window.open && display) {
        // process user input
        glfwPollEvents();

        // Dear ImGui frame setup
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Set up UI components
        auto flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;
        float  padding      = 0.0f;
        ImVec2 bottom_right = ImVec2(ImGui::GetIO().DisplaySize.x - padding, ImGui::GetIO().DisplaySize.y - padding);
        ImGui::SetNextWindowPos(bottom_right, ImGuiCond_Always, ImVec2(1.0, 1.0));
        ImGui::Begin("Frame time", NULL, flags);
        ImGui::Text("Particles: %i\nAvg. time: %.3f ms (%.2f%% compute) ", pc.numParticles, avgTimeCompute,
                    avgTimeCompute / avgTimeTotal * 100);
        ImGui::End();
        // ImGui::ShowDemoWindow();

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
            pc.removeFlaggedParticles();
            stopCompute.record();

            // Copy back to CPU
            pc.copyToCPU();

            stopCopy.record();

            float elapsedCompute, elapsedCopy;
            elapsedCompute = cuda::eventElapsedTime(start, stopCompute);
            elapsedCopy    = cuda::eventElapsedTime(start, stopCopy);

            avgTimeCompute = (1 - timeConst) * avgTimeCompute + timeConst * elapsedCompute;
            avgTimeTotal   = (1 - timeConst) * avgTimeTotal + timeConst * elapsedCopy;
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
            // this is pretty inefficient, as we have to copy a lot of identical vertex and normal data over to the
            // GPU for each particle Ideally, we'd use instancing to do better, and only transfer the model matrix
            // over at each timestep see https://learnopengl.com/Advanced-OpenGL/Instancing
            Transform t;
            t.scale     = particleScale;
            t.translate = glm::vec3{pc.position[i].x, pc.position[i].y, pc.position[i].z};
            t.color     = pc.weight[i] > 0 ? particleColor : particleColorOOB;
            particleMesh.draw(shader, t);
        }

        // ImGui::Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        window.checkForUpdates();
        frame += 1;
    }

    // Shut down ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return 0;
}
