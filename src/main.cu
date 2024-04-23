// C++ headers
#include <iostream>
#include <string>
#include <vector>

// GLM headers
#include <glm/glm.hpp>

// ImGUI headers
#include "imgui.h"

// My headers (c++)
#include "app.hpp"
#include "input.hpp"
#include "mesh.hpp"
#include "shader.hpp"
#include "surface.hpp"
#include "window.hpp"

// My headers (CUDA)
#include "cuda.cuh"
#include "particle_container.cuh"
#include "triangle.cuh"

using std::vector, std::string;

string printTime (double time_s) {
    char   buf[64];
    int    factor = 1;
    string str    = "s";

    if (time_s < 1e-6) {
        factor = 1'000'000'000;
        str    = "ns";
    } else if (time_s < 1e-3) {
        factor = 1'000'000;
        str    = "us";
    } else if (time_s < 1) {
        factor = 1000;
        str    = "ms";
    }

    sprintf(buf, "%.3f %s", time_s * factor, str.c_str());

    return {buf};
}

int main (int argc, char *argv[]) {
    // Handle command line arguments
    string filename("../input.toml");
    if (argc > 1) {
        filename = argv[1];
    }

    Window window("Sputterer", app::SCR_WIDTH, app::SCR_HEIGHT);

    glfwSetFramebufferSizeCallback(window.window, app::framebufferSizeCallback);
    glfwSetCursorPosCallback(window.window, app::mouseCursorCallback);
    glfwSetScrollCallback(window.window, app::scrollCallback);

    Shader shader("../shaders/shader.vert", "../shaders/shader.frag");
    shader.use();

    Input input(filename);
    input.read();

    std::cout << "Input read." << std::endl;

    app::camera.orientation = glm::normalize(glm::vec3(input.chamberRadius));
    app::camera.distance    = 2.0f * input.chamberRadius;
    app::camera.yaw         = -135;
    app::camera.pitch       = 30;
    app::camera.updateVectors();

    // Create particle container, including any explicitly-specified initial particles
    ParticleContainer pc{"noname", 1.0f, 1};
    pc.addParticles(input.particle_x, input.particle_y, input.particle_z, input.particle_vx, input.particle_vy,
                    input.particle_vz, input.particle_w);

    vec3 particleColor{0.05f};
    vec3 particleColorOOB{1.0f, 0.2f, 0.2f};
    vec3 particleScale{0.01f};

    // Read mesh from file
    Mesh particleMesh{};
    particleMesh.readFromObj("../o_sphere.obj");
    particleMesh.setBuffers();

    // construct triangles
    host_vector<Triangle> h_triangles;
    host_vector<size_t>   h_materialIDs;
    host_vector<Material> h_materials;
    host_vector<char>     h_to_collect;
    std::vector<int>      collect_inds;
    std::vector<string>   surfaceNames;

    for (size_t id = 0; id < input.surfaces.size(); id++) {
        const auto &surf     = input.surfaces.at(id);
        const auto &mesh     = surf.mesh;
        const auto &material = surf.material;

        surfaceNames.push_back(surf.name);
        h_materials.push_back(surf.material);

        for (const auto &[i1, i2, i3] : mesh.triangles) {
            auto model = surf.transform.getMatrix();
            auto v1    = make_float3(model * glm::vec4(mesh.vertices[i1].pos, 1.0));
            auto v2    = make_float3(model * glm::vec4(mesh.vertices[i2].pos, 1.0));
            auto v3    = make_float3(model * glm::vec4(mesh.vertices[i3].pos, 1.0));

            h_triangles.push_back({v1, v2, v3});
            h_materialIDs.push_back(id);
            if (material.collect) {
                collect_inds.push_back(static_cast<int>(h_triangles.size()) - 1);
            }
        }
    }

    host_vector<int> collected(collect_inds.size(), 0);

    std::cout << "Meshes read." << std::endl;

    // Send mesh data to GPU. Really slow for some reason (multiple seconds)!
    device_vector<Triangle> d_triangles{h_triangles};
    device_vector<size_t>   d_surfaceIDs{h_materialIDs};
    device_vector<Material> d_materials{h_materials};
    device_vector<int>      d_collected(h_triangles.size(), 0);

    std::cout << "Mesh data sent to GPU." << std::endl;

    // Create timing objects
    size_t frame = 0;

    float  avgTimeCompute = 0.0f, avgTimeTotal = 0.0f;
    float  iterReset    = 100;
    float  timeConst    = 1 / iterReset;
    double physicalTime = 0, physicalTimestep = 0;

    cuda::event start{}, stopCompute{}, stopCopy{};

    std::cout << "Beginning main loop." << std::endl;

    while (window.open) {

        Window::beginRenderLoop();

        // Timing info
        auto flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;
        float  padding      = 0.0f;
        ImVec2 bottom_right = ImVec2(ImGui::GetIO().DisplaySize.x - padding, ImGui::GetIO().DisplaySize.y - padding);
        ImGui::SetNextWindowPos(bottom_right, ImGuiCond_Always, ImVec2(1.0, 1.0));
        ImGui::Begin("Frame time", nullptr, flags);
        ImGui::Text("Simulation timestep: %s\nSimulation time: %s\nCompute time: %.3f ms (%.2f%% data transfer)  "
                    "\nParticles: %i",
                    printTime(physicalTimestep).c_str(), printTime(physicalTime).c_str(), avgTimeCompute,
                    (1.0f - avgTimeCompute / avgTimeTotal) * 100, pc.numParticles);
        ImGui::End();

        // Table of collected particle amounts
        auto   tableFlags  = ImGuiTableFlags_BordersH;
        ImVec2 bottom_left = ImVec2(0, ImGui::GetIO().DisplaySize.y - padding);
        ImGui::SetNextWindowPos(bottom_left, ImGuiCond_Always, ImVec2(0.0, 1.0));
        ImGui::Begin("Particle collection info", nullptr, flags);
        if (ImGui::BeginTable("Table", 3, tableFlags)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Surface name");
            ImGui::TableNextColumn();
            ImGui::Text("Triangle ID");
            ImGui::TableNextColumn();
            ImGui::Text("Collection rate (#/s)");
            for (int row = 0; row < collect_inds.size(); row++) {
                auto triangleID = collect_inds[row];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", surfaceNames.at(h_materialIDs[triangleID]).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%i", static_cast<int>(triangleID));
                ImGui::TableNextColumn();
                ImGui::Text("%.3e", static_cast<double>(collected[row]) / physicalTime);
            }
            ImGui::EndTable();
        }
        ImGui::End();

        // frame timing for rendering
        auto currentFrame = static_cast<float>(glfwGetTime());
        app::deltaTime    = currentFrame - app::lastFrame;
        app::lastFrame    = currentFrame;
        app::processInput(window.window);

        auto thisTimestep = input.timestep * app::deltaTime;
        physicalTime += thisTimestep;
        physicalTimestep = (1 - timeConst) * physicalTimestep + timeConst * input.timestep * app::deltaTime;

        // record compute start time
        if (frame > 1) {
            start.record();

            // Emit particles
            size_t triCount{0};
            for (const auto &surf : input.surfaces) {
                auto &emitter = surf.emitter;
                if (!emitter.emit) {
                    continue;
                }

                for (size_t i = 0; i < surf.mesh.numTriangles; i++) {
                    pc.emit(h_triangles[i], emitter, physicalTimestep);
                }
                triCount += surf.mesh.numTriangles;
            }

            // Push particles
            pc.push(physicalTimestep, d_triangles, d_surfaceIDs, d_materials, d_collected);

            // Remove particles that are out of bounds
            pc.flagOutOfBounds(input.chamberRadius, input.chamberLength);
            pc.removeFlaggedParticles();
            stopCompute.record();

            // Track particles collected by each triangle flagged 'collect'
            for (int id = 0; id < collect_inds.size(); id++) {
                auto oldVal  = collected[id];
                auto d_begin = d_collected.begin() + collect_inds[id];
                thrust::copy(d_begin, d_begin + 1, collected.begin() + id);
                collected[id] += oldVal;
            }

            // Copy particle data back to CPU
            pc.copyToCPU();

            stopCopy.record();

            float elapsedCompute, elapsedCopy;
            elapsedCompute = cuda::eventElapsedTime(start, stopCompute);
            elapsedCopy    = cuda::eventElapsedTime(start, stopCopy);

            avgTimeCompute = (1 - timeConst) * avgTimeCompute + timeConst * elapsedCompute;
            avgTimeTotal   = (1 - timeConst) * avgTimeTotal + timeConst * elapsedCopy;
        }

        // update camera projection
        shader.updateView(app::camera, app::aspectRatio);

        for (const auto &surface : input.surfaces) {
            // set the model matrix
            surface.mesh.draw(shader, surface.transform, surface.color);
        }

        for (int i = 0; i < pc.numParticles; i++) {
            // this is pretty inefficient, as we have to copy a lot of identical vertex and normal data over to the
            // GPU for each particle Ideally, we'd use instancing to do better, and only transfer the model matrix
            // over at each timestep see https://learnopengl.com/Advanced-OpenGL/Instancing
            Transform t;
            t.scale     = particleScale;
            t.translate = glm::vec3{pc.position[i].x, pc.position[i].y, pc.position[i].z};
            auto color  = pc.weight[i] > 0 ? particleColor : particleColorOOB;
            particleMesh.draw(shader, t, color);
        }

        window.endRenderLoop();
        frame += 1;
    }

    std::cout << "Program terminated successfully." << std::endl;

    return 0;
}
