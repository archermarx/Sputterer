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
#include "Input.hpp"
#include "Shader.hpp"
#include "Surface.hpp"
#include "Window.hpp"
#include "ThrusterPlume.hpp"
#include "Constants.hpp"
#include "Output.hpp"

// My headers (CUDA)
#include "cuda.cuh"
#include "ParticleContainer.cuh"
#include "Triangle.cuh"

using std::vector, std::string;

struct DepositionInfo {
    size_t num_tris = 0;
    vector<string> surf_names;
    vector<size_t> local_indices;
    vector<size_t> global_indices;
    vector<double> areas;
    vector<int> particles_collected;
    vector<double> deposition_rates;
    vector<double> mass_fluxes;
};

int main (int argc, char *argv[]) {
    using namespace constants;
    // Initialize GPU
    device_vector<int> init{0};

    // Read input and open window, if required
    std::string filename = argc > 1 ? argv[1] : "input.toml";
    Input input = read_input(filename);
    auto window = app::initialize(input);

    // construct triangles and set up diagnostics
    host_vector<Triangle> h_triangles;
    host_vector<size_t> h_material_ids;
    host_vector<Material> h_materials;
    DepositionInfo deposition_info{};
    auto &geometry = input.geometry;

    for (size_t id = 0; id < geometry.surfaces.size(); id++) {
        const auto &surf = geometry.surfaces.at(id);
        const auto &mesh = surf.mesh;
        const auto &material = surf.material;

        h_materials.push_back(surf.material);

        auto ind = 0;
        for (const auto &[i1, i2, i3]: mesh.triangles) {
            auto model = surf.transform.get_matrix();
            auto v1 = make_float3(model*glm::vec4(mesh.vertices[i1].pos, 1.0));
            auto v2 = make_float3(model*glm::vec4(mesh.vertices[i2].pos, 1.0));
            auto v3 = make_float3(model*glm::vec4(mesh.vertices[i3].pos, 1.0));

            Triangle tri{v1, v2, v3};
            h_triangles.push_back(tri);
            h_material_ids.push_back(id);
            if (material.collect) {
                auto global_index = static_cast<int>(h_triangles.size()) - 1;
                Triangle tri{v1, v2, v3};
                deposition_info.surf_names.push_back(surf.name);
                deposition_info.areas.push_back(tri.area);
                deposition_info.global_indices.push_back(global_index);
                deposition_info.local_indices.push_back(ind);
                deposition_info.particles_collected.push_back(0);
                deposition_info.deposition_rates.push_back(0);
                deposition_info.mass_fluxes.push_back(0);
                deposition_info.num_tris++;
            }
            ind++;
        }
    }

    if (input.verbosity > 0) std::cout << "Meshes read." << std::endl;

    // Construct BVH on CPU
    host_vector<BVHNode> h_nodes;
    host_vector<size_t> h_triangle_indices;
    Scene h_scene;
    h_scene.build(h_triangles, h_triangle_indices, h_nodes);

    if (input.verbosity > 0) std::cout << "Bounding volume heirarchy constructed." << std::endl;

    // Send mesh data and BVH to GPU.
    device_vector<Triangle> d_triangles = h_triangles;
    device_vector<size_t> d_surface_ids{h_material_ids};
    device_vector<Material> d_materials{h_materials};
    device_vector<int> d_collected(h_triangles.size(), 0);

    Scene d_scene(h_scene);
    device_vector<BVHNode> d_nodes = h_nodes;
    device_vector<size_t> d_triangle_indices = h_triangle_indices;
    d_scene.triangles = thrust::raw_pointer_cast(d_triangles.data());
    d_scene.triangle_indices = thrust::raw_pointer_cast(d_triangle_indices.data());
    d_scene.nodes = thrust::raw_pointer_cast(d_nodes.data());

    if (input.verbosity > 0) std::cout << "Mesh data sent to GPU" << std::endl;

    // Cast initial rays from plume to find where they hit facility geometry
    // Store result in ParticleContainer pc_plume
    host_vector<HitInfo> hits;
    host_vector<float3> hit_positions;
    host_vector<float> num_emit;
    auto plume = input.plume;
    plume.find_hits(input, h_scene, h_materials, h_material_ids, hits, hit_positions, num_emit);

    // Copy plume results to GPU
    device_vector<HitInfo> d_hits{hits};
    device_vector<float> d_num_emit{num_emit};

    // Create particle container for carbon atoms and renderer
    ParticleContainer particles{"carbon", max_particles, 1.0f, 1};
    app::Renderer renderer(input, &h_scene, plume, particles, geometry);

    // Create timing objects
    size_t step = 0;
    app::Timer timer;
    cuda::Event start{}, stop_compute{}, stop_copy{};

    // Create output file for deposition
    Output output("deposition.csv");

    if (input.verbosity > 0) std::cout << "Beginning main loop." << std::endl;

    while ((input.display && window.open) || (!input.display && timer.physical_time < input.max_time_s)) {
        // Draw GUI and set up for this frame
        app::begin_frame(step, input, window, renderer, timer);

        // TODO: can we move this out of main into a different function
        if (input.display) {
            // Table of collected particle amounts
            auto table_flags = ImGuiTableFlags_BordersH;
            ImVec2 bottom_left = ImVec2(0, ImGui::GetIO().DisplaySize.y);
            ImGui::SetNextWindowPos(bottom_left, ImGuiCond_Always, ImVec2(0.0, 1.0));
            ImGui::Begin("Particle collection info", nullptr, app::imgui_flags);
            if (ImGui::BeginTable("Table", 4, table_flags)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Surface name");
                ImGui::TableNextColumn();
                ImGui::Text("Triangle ID");
                ImGui::TableNextColumn();
                ImGui::Text("Particles collected");
                ImGui::TableNextColumn();
                ImGui::Text("Deposition rate [um/kh]");
                for (int tri = 0; tri < deposition_info.num_tris; tri++) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", deposition_info.surf_names[tri].c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%i", deposition_info.local_indices[tri]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", deposition_info.particles_collected[tri]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", deposition_info.deposition_rates[tri]);
                }
                ImGui::EndTable();
            }
            ImGui::End();
        }

        // Main computation loop
        if (step > 0 && !app::sim_paused) {
            start.record();

            // Push particles and sputter from surfaces
            particles.evolve(d_scene, d_materials, d_surface_ids, d_collected,
                             d_hits, d_num_emit, input.particle_weight,
                             input.timestep_s);

            // flag particles that are out of bounds
            particles.flag_out_of_bounds(input.chamber_radius_m, input.chamber_length_m);

            // remove particles with negative weight (out of bounds and phantom emitted particles)
            particles.remove_flagged_particles();

            // record stop time
            stop_compute.record();

            // Track particles collected by each triangle flagged 'collect' and compute diagnostics
            for (int id = 0; id < deposition_info.num_tris; id++) {
                // Copy number of particles collected to CPU
                auto d_begin = d_collected.begin() + deposition_info.global_indices[id];
                thrust::copy(d_begin, d_begin + 1, deposition_info.particles_collected.begin() + id);

                // Compute deposition rate and carbon flux
                double mass_carbon = deposition_info.particles_collected[id]*input.particle_weight*carbon.mass*m_u;
                double volume_carbon = mass_carbon/graphite_density;
                double triangle_area = deposition_info.areas[id];
                double layer_thickness_um = volume_carbon/triangle_area*1e6;
                double physical_time_kh = timer.physical_time/3600/1000;
                deposition_info.deposition_rates[id] = layer_thickness_um/physical_time_kh;
                deposition_info.mass_fluxes[id] = mass_carbon / triangle_area / timer.physical_time;
            }

            // Copy particle data back to CPU
            particles.copy_to_cpu();
            stop_copy.record();

            // timing
            double elapsed_compute = cuda::event_elapsed_time(start, stop_compute);
            double elapsed_copy = cuda::event_elapsed_time(start, stop_copy);
            timer.update_averages(elapsed_compute, elapsed_copy);
        }

        // Draw scene
        renderer.draw(input);

        // Finalize frame and increment timestep
        app::end_frame(input, window);
        if (!app::sim_paused) {
            step ++;
            timer.physical_time += input.timestep_s;
        }

        // Write output to console and file at regular intervals, plus one additional when simulation terminates
        if ((!app::sim_paused && timer.should_output()) ||
            (!input.display && timer.physical_time >= input.max_time_s) ||
            (input.display && !window.open)) {

            std::cout << "  Step " << step
                      << ", Simulation time: " << app::print_time(timer.physical_time)
                      << ", Timestep: " << app::print_time(input.timestep_s)
                      << ", Avg. step time: " << timer.dt_smoothed << " ms" << std::endl;

            timer.next_output_time += input.output_interval_s;
        }
    }

    if (input.verbosity > 0) std::cout << "Program terminated successfully." << std::endl;

    return 0;
}
