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

string print_time (double time_s) {
    char buf[64];
    int factor = 1;
    string str = "s";

    if (time_s < 1e-6) {
        factor = 1'000'000'000;
        str = "ns";
    } else if (time_s < 1e-3) {
        factor = 1'000'000;
        str = "us";
    } else if (time_s < 1) {
        factor = 1000;
        str = "ms";
    }

    sprintf(buf, "%.3f %s", time_s*factor, str.c_str());

    return {buf};
}


int main (int argc, char *argv[]) {
    using namespace constants;
    // Initialize GPU
    device_vector<int> init{0};

    // Read input and open window, if required
    std::string filename = argc > 1 ? argv[1] : "input.toml";
    Input input = read_input(filename);
    auto window = app::initialize(input);

    // construct triangles
    // TODO: move to a function, maybe within input or scene geometry
    host_vector<Triangle> h_triangles;
    host_vector<size_t> h_material_ids;
    host_vector<Material> h_materials;
    host_vector<char> h_to_collect;
    std::vector<int> collect_inds_global;
    std::vector<int> collect_inds_local;
    std::vector<string> surface_names;
    auto &geometry = input.geometry;

    for (size_t id = 0; id < geometry.surfaces.size(); id++) {
        const auto &surf = geometry.surfaces.at(id);
        const auto &mesh = surf.mesh;
        const auto &material = surf.material;

        surface_names.push_back(surf.name);
        h_materials.push_back(surf.material);

        auto ind = 0;
        for (const auto &[i1, i2, i3]: mesh.triangles) {
            auto model = surf.transform.get_matrix();
            auto v1 = make_float3(model*glm::vec4(mesh.vertices[i1].pos, 1.0));
            auto v2 = make_float3(model*glm::vec4(mesh.vertices[i2].pos, 1.0));
            auto v3 = make_float3(model*glm::vec4(mesh.vertices[i3].pos, 1.0));

            h_triangles.push_back({v1, v2, v3});
            h_material_ids.push_back(id);
            if (material.collect) {
                collect_inds_global.push_back(static_cast<int>(h_triangles.size()) - 1);
                collect_inds_local.push_back(ind);
            }
            ind++;
        }
    }
    std::vector<double> deposition_rates(collect_inds_global.size(), 0);
    host_vector<int> collected(collect_inds_global.size(), 0);

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

    // Create particle container for carbon atoms and renderer for BVH
    ParticleContainer particles{"carbon", max_particles, 1.0f, 1};
    BVHRenderer bvh(&h_scene);
    app::Renderer renderer = {bvh, plume, particles, geometry};
    renderer.setup(input);

    // Create timing objects
    size_t step = 0;
    float avg_time_compute = 0.0f, avg_time_total = 0.0f;
    float iter_reset = 25;
    float time_const = 1/iter_reset;
    double physical_time = 0;
    float delta_time_smoothed = 0;

    auto next_output_time = 0.0f;

    cuda::Event start{}, stop_compute{}, stop_copy{};

    auto current_time = std::chrono::system_clock::now();
    auto last_time = std::chrono::system_clock::now();

    // Create output file for deposition
    Output output("deposition.csv");

    if (input.verbosity > 0) {
        std::cout << "Beginning main loop." << std::endl;
    }

    // Pause simulation if displaying
    app::sim_paused = input.display;

    while ((input.display && window.open) || (!input.display && physical_time < input.max_time_s)) {
        // TODO: can we move this out of main into a different function
        if (input.display) {
            Window::begin_render_loop();

            // Timing info
            auto flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;
            float padding = 0.0f;
            ImVec2 bottom_right = ImVec2(ImGui::GetIO().DisplaySize.x - padding, ImGui::GetIO().DisplaySize.y - padding);
            ImGui::SetNextWindowPos(bottom_right, ImGuiCond_Always, ImVec2(1.0, 1.0));
            ImGui::Begin("Frame time", nullptr, flags);
            ImGui::Text("Simulation step %li (%s)\nSimulation time: %s\nCompute time: %.3f ms (%.2f%% data "
                    "transfer)   \nFrame time: %.3f ms (%.1f fps, %.2f%% compute)   \nParticles: %i", step, print_time(
                        input.timestep_s).c_str(), print_time(physical_time).c_str(), avg_time_compute,
                    (1.0f - avg_time_compute/avg_time_total)*100, delta_time_smoothed, 1000/delta_time_smoothed,
                    (avg_time_total/delta_time_smoothed)*100, particles.num_particles);
            ImGui::End();

            // Table of collected particle amounts
            auto table_flags = ImGuiTableFlags_BordersH;
            ImVec2 bottom_left = ImVec2(0, ImGui::GetIO().DisplaySize.y - padding);
            ImGui::SetNextWindowPos(bottom_left, ImGuiCond_Always, ImVec2(0.0, 1.0));
            ImGui::Begin("Particle collection info", nullptr, flags);
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
                for (int row = 0; row < collect_inds_global.size(); row++) {

                    auto triangle_id_global = collect_inds_global[row];
                    double mass_carbon = collected[row]*input.particle_weight*carbon.mass*m_u;
                    double volume_carbon = mass_carbon/graphite_density;
                    double triangle_area = h_triangles[triangle_id_global].area;
                    double layer_thickness_um = volume_carbon/triangle_area*1e6;
                    double physical_time_kh = physical_time/3600/1000;
                    double deposition_rate = layer_thickness_um/physical_time_kh;
                    deposition_rates[row] = deposition_rate;

                    auto triangle_id = collect_inds_global[row];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", surface_names.at(h_material_ids[triangle_id]).c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%i", static_cast<int>(collect_inds_local[row]));
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", collected[row]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", deposition_rates[row]);
                }
                ImGui::EndTable();
            }
            ImGui::End();

            flags = ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_AlwaysAutoResize |
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoSavedSettings;

            ImVec2 top_right = ImVec2(ImGui::GetIO().DisplaySize.x - padding, 0);
            ImGui::SetNextWindowPos(top_right, ImGuiCond_Always, ImVec2(1.0, 0.0));
            ImGui::Begin("Options", nullptr, flags);
            if (ImGui::BeginTable("split", 1)) {
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show plume cone", &plume.render);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show plume particles", &plume.particles.render);
                ImGui::TableNextColumn();
                ImGui::Text("Plume particle scale");
                ImGui::TableNextColumn();
                ImGui::SliderFloat("##plume_particle_scale", &plume.particles.scale, 0, 0.3);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show sputtered particles", &particles.render);
                ImGui::TableNextColumn();
                ImGui::Text("Sputtered particle scale");
                ImGui::TableNextColumn();
                ImGui::SliderFloat("##sputtered_particle_scale", &particles.scale, 0, 0.3);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show bounding boxes", &bvh.render);
                ImGui::TableNextColumn();
                ImGui::Text("Bounding box depth");
                ImGui::TableNextColumn();
                ImGui::SliderInt("##bvh_depth", &bvh.draw_depth, 0, h_scene.bvh_depth);
            }
            ImGui::EndTable();
            ImGui::End();
        }

        // Record iteration timing information
        current_time = std::chrono::system_clock::now();
        app::delta_time =
            static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time).count())/
            1e6;
        last_time = current_time;

        // set physical timestep_s. if we're displaying a window, we set the physical timestep_s based on the rendering
        // timestep_s to get smooth performance at different window sizes. If not, we just use the user-provided timestep_s
        if (!app::sim_paused) {
            physical_time += input.timestep_s;
        }
        delta_time_smoothed = (1 - time_const)*delta_time_smoothed + time_const*app::delta_time*1000;

        // Main computations
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

            // Track particles collected by each triangle flagged 'collect'
            for (int id = 0; id < collect_inds_global.size(); id++) {
                auto d_begin = d_collected.begin() + collect_inds_global[id];
                thrust::copy(d_begin, d_begin + 1, collected.begin() + id);
            }

            // Copy particle data back to CPU
            particles.copy_to_cpu();
            stop_copy.record();

            // timing
            float elapsed_compute, elapsed_copy;
            elapsed_compute = cuda::event_elapsed_time(start, stop_compute);
            elapsed_copy = cuda::event_elapsed_time(start, stop_copy);

            avg_time_compute = (1 - time_const)*avg_time_compute + time_const*elapsed_compute;
            avg_time_total = (1 - time_const)*avg_time_total + time_const*elapsed_copy;
        }

        renderer.draw(input);

        if (!app::sim_paused && physical_time > next_output_time ||
                (!input.display && physical_time >= input.max_time_s) ||
                (input.display && !window.open)) {
            // Write output to console at regular intervals, plus one additional when simulation terminates
            std::cout << "Step " << step << ", Simulation time: " << print_time(physical_time)
                << ", Timestep: " << print_time(input.timestep_s) << ", Avg. step time: " << delta_time_smoothed
                << " ms" << std::endl;

            // write output to file
            next_output_time += input.output_interval_s;}

        app::end_frame(input, window);
        step += !app::sim_paused;
    }

    if (input.verbosity > 0) std::cout << "Program terminated successfully." << std::endl;

    return 0;
}
