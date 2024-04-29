// C++ headers
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

// GLM headers
#include <glm/glm.hpp>

// ImGUI headers
#include "imgui.h"

// My headers (c++)
#include "app.hpp"
#include "input.hpp"
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
    string filename{"../input.toml"};
    bool   display{true};

    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        display = static_cast<bool>(std::stoi(argv[2]));
    }

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

    // construct triangles
    host_vector<Triangle> h_triangles;

    host_vector<size_t>   h_materialIDs;
    host_vector<Material> h_materials;

    host_vector<char> h_to_collect;
    std::vector<int>  collect_inds_global;
    std::vector<int>  collect_inds_local;

    std::vector<string> surfaceNames;

    for (size_t id = 0; id < input.surfaces.size(); id++) {
        const auto &surf     = input.surfaces.at(id);
        const auto &mesh     = surf.mesh;
        const auto &material = surf.material;

        surfaceNames.push_back(surf.name);
        h_materials.push_back(surf.material);

        auto ind = 0;
        for (const auto &[i1, i2, i3] : mesh.triangles) {
            auto model = surf.transform.getMatrix();
            auto v1    = make_float3(model * glm::vec4(mesh.vertices[i1].pos, 1.0));
            auto v2    = make_float3(model * glm::vec4(mesh.vertices[i2].pos, 1.0));
            auto v3    = make_float3(model * glm::vec4(mesh.vertices[i3].pos, 1.0));

            h_triangles.push_back({v1, v2, v3});
            h_materialIDs.push_back(id);
            if (material.collect) {
                collect_inds_global.push_back(static_cast<int>(h_triangles.size()) - 1);
                collect_inds_local.push_back(ind);
            }
            ind++;
        }
    }

    host_vector<int> collected(collect_inds_global.size(), 0);
    std::cout << "Meshes read." << std::endl;

    // Send mesh data to GPU. Really slow for some reason (multiple seconds)!
    device_vector<Triangle> d_triangles = h_triangles;
    device_vector<size_t>   d_surfaceIDs{h_materialIDs};
    device_vector<Material> d_materials{h_materials};
    device_vector<int>      d_collected(h_triangles.size(), 0);

    // Display objects
    Window window{.name = "Sputterer", .width = app::SCR_WIDTH, .height = app::SCR_HEIGHT};
    Shader meshShader{}, particleShader{};
    if (display) {
        // enable window
        window.enable();

        // Register window callbacks
        glfwSetFramebufferSizeCallback(window.window, app::framebufferSizeCallback);
        glfwSetCursorPosCallback(window.window, app::mouseCursorCallback);
        glfwSetScrollCallback(window.window, app::scrollCallback);

        // Load mesh shader
        meshShader.load("../shaders/shader.vert", "../shaders/shader.frag");

        // initialize mesh buffers
        for (auto &surf : input.surfaces) {
            surf.mesh.setBuffers();
        }

        // Load particle shader
        particleShader.load("../shaders/particle.vert", "../shaders/particle.frag");
        particleShader.use();
        constexpr vec3 particleColor{0.05f};
        constexpr vec3 particleScale{0.01f};
        particleShader.setVec3("scale", particleScale);
        particleShader.setVec3("objectColor", particleColor);

        // Set up particle mesh
        pc.mesh.readFromObj("../o_sphere.obj");
        pc.setBuffers();
    }

    // Create timing objects
    size_t frame = 0;

    float  avgTimeCompute = 0.0f, avgTimeTotal = 0.0f;
    float  iterReset    = 25;
    float  timeConst    = 1 / iterReset;
    double physicalTime = 0, physicalTimestep = 0;
    float  deltaTimeSmoothed = 0;

    auto nextOutputTime = 0.0f;

    cuda::event start{}, stopCompute{}, stopCopy{};

    std::cout << "Beginning main loop." << std::endl;

    auto current_time = std::chrono::system_clock::now();
    auto last_time    = std::chrono::system_clock::now();

    // Create output file for deposition
    string        output_filename{"deposition.csv"};
    std::ofstream output_file;
    output_file.open(output_filename);
    output_file << "Time(s),Surface name,Local triangle ID,Global triangle ID,Particles collected" << std::endl;
    output_file.close();

    while ((display && window.open) || (!display && physicalTime < input.max_time)) {

        if (display) {
            Window::beginRenderLoop();

            // Timing info
            auto flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;
            float  padding = 0.0f;
            ImVec2 bottom_right =
                ImVec2(ImGui::GetIO().DisplaySize.x - padding, ImGui::GetIO().DisplaySize.y - padding);
            ImGui::SetNextWindowPos(bottom_right, ImGuiCond_Always, ImVec2(1.0, 1.0));
            ImGui::Begin("Frame time", nullptr, flags);
            ImGui::Text("Simulation step %li (%s)\nSimulation time: %s\nCompute time: %.3f ms (%.2f%% data "
                        "transfer)   \nFrame time: %.3f ms (%.1f fps, %.2f%% compute)   \nParticles: %i",
                        frame, printTime(physicalTimestep).c_str(), printTime(physicalTime).c_str(), avgTimeCompute,
                        (1.0f - avgTimeCompute / avgTimeTotal) * 100, deltaTimeSmoothed, 1000 / deltaTimeSmoothed,
                        (avgTimeTotal / deltaTimeSmoothed) * 100, pc.numParticles);
            ImGui::End();

            // Table of collected particle amounts
            auto   tableFlags  = ImGuiTableFlags_BordersH;
            ImVec2 bottom_left = ImVec2(0, ImGui::GetIO().DisplaySize.y - padding);
            ImGui::SetNextWindowPos(bottom_left, ImGuiCond_Always, ImVec2(0.0, 1.0));
            ImGui::Begin("Particle collection info", nullptr, flags);
            if (ImGui::BeginTable("Table", 4, tableFlags)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Surface name");
                ImGui::TableNextColumn();
                ImGui::Text("Triangle ID");
                ImGui::TableNextColumn();
                ImGui::Text("Particles collected");
                ImGui::TableNextColumn();
                ImGui::Text("Collection rate (#/s)");
                for (int row = 0; row < collect_inds_global.size(); row++) {
                    auto triangleID = collect_inds_global[row];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", surfaceNames.at(h_materialIDs[triangleID]).c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%i", static_cast<int>(collect_inds_local[row]));
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", collected[row]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3e", static_cast<double>(collected[row]) / physicalTime);
                }
                ImGui::EndTable();
            }
            ImGui::End();
        }

        // Record iteration timing information
        current_time   = std::chrono::system_clock::now();
        app::deltaTime = static_cast<double>(
                             std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time).count()) /
                         1e6;
        last_time = current_time;

        // set physical timestep. if we're displaying a window, we set the physical timestep based on the rendering
        // timestep to get smooth performance at different window sizes. If not, we just use the user-provided timestep
        float thisTimestep{0};
        if (display) {
            thisTimestep = input.timestep * app::deltaTime / (15e-3);
        } else {
            thisTimestep = input.timestep;
        }
        physicalTime += thisTimestep;
        physicalTimestep  = (1 - timeConst) * physicalTimestep + timeConst * thisTimestep;
        deltaTimeSmoothed = (1 - timeConst) * deltaTimeSmoothed + timeConst * app::deltaTime * 1000;

        // Main computations
        if (frame > 0) {
            start.record();

            // Emit particles
            size_t triCount{0};
            for (const auto &surf : input.surfaces) {
                auto &emitter = surf.emitter;
                if (!emitter.emit) {
                    continue;
                }

                for (size_t i = 0; i < surf.mesh.numTriangles; i++) {
                    pc.emit(h_triangles[i], emitter, thisTimestep);
                }
                triCount += surf.mesh.numTriangles;
            }

            // Push particles
            pc.push(thisTimestep, d_triangles, d_surfaceIDs, d_materials, d_collected);

            // Remove particles that are out of bounds
            pc.flagOutOfBounds(input.chamberRadius, input.chamberLength);
            pc.removeFlaggedParticles();
            stopCompute.record();

            // Track particles collected by each triangle flagged 'collect'
            for (int id = 0; id < collect_inds_global.size(); id++) {
                auto d_begin = d_collected.begin() + collect_inds_global[id];
                thrust::copy(d_begin, d_begin + 1, collected.begin() + id);
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

        // Rendering
        if (display) {
            // update camera projection mesh shader
            meshShader.use();
            meshShader.updateView(app::camera, app::aspectRatio);

            // draw meshes
            for (const auto &surface : input.surfaces) {
                // set the model matrix
                meshShader.use();
                surface.mesh.draw(meshShader, surface.transform, surface.color);
            }

            // draw particles (instanced!)
            if (pc.numParticles > 0) {
                // activate particle shader
                particleShader.use();

                // send camera information to shader
                auto cam = app::camera.getProjectionMatrix(app::aspectRatio) * app::camera.getViewMatrix();
                particleShader.setMat4("camera", cam);

                // draw particles
                pc.draw(particleShader);
            }
        }

        if (physicalTime > nextOutputTime || (!display && physicalTime >= input.max_time) ||
            (display && !window.open)) {
            // Write output to console at regular intervals, plus one additional when simulation terminates
            std::cout << "Step " << frame << ", Simulation time: " << printTime(physicalTime)
                      << ", Timestep: " << printTime(physicalTimestep) << ", Avg. step time: " << deltaTimeSmoothed
                      << " ms" << std::endl;

            // Log deposition rate info
            output_file.open(output_filename, std::ios_base::app);
            for (int i = 0; i < collect_inds_global.size(); i++) {
                auto triangle_id_global = collect_inds_global[i];
                output_file << physicalTime << ",";
                output_file << surfaceNames.at(h_materialIDs[triangle_id_global]) << ",";
                output_file << collect_inds_local.at(i) << ",";
                output_file << triangle_id_global << ",";
                output_file << collected[i] << "\n";
            }
            output_file.close();

            nextOutputTime += input.output_interval;
        }

        if (display) {
            window.endRenderLoop();
            app::processInput(window.window);
        }

        frame += 1;
    }

    std::cout << "Program terminated successfully." << std::endl;

    return 0;
}
