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
#include "app.h"
#include "Input.h"
#include "Shader.h"
#include "Surface.h"
#include "Window.h"
#include "ThrusterPlume.h"
#include "Timer.h"
#include "Constants.h"
#include "DepositionInfo.h"
#include "Renderer.h"
#include "cuda_helpers.h"
#include "ParticleContainer.h"
#include "Triangle.h"

using std::vector, std::string;

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

    std::vector<Surface>& surfaces = input.surfaces;

    DepositionInfo deposition_info("deposition.csv");

    // TODO: can this be simplified and moved into a function?
    for (size_t id = 0; id < surfaces.size(); id++) {
        const auto &surf = surfaces.at(id);
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
                deposition_info.surface_names.push_back(surf.name);
                deposition_info.triangle_areas.push_back(tri.area);
                auto c = tri.centroid;
                deposition_info.centroid_x.push_back(c.x);
                deposition_info.centroid_y.push_back(c.y);
                deposition_info.centroid_z.push_back(c.z);
                deposition_info.global_indices.push_back(global_index);
                deposition_info.local_indices.push_back(ind);
                deposition_info.num_tris++;
            }
            ind++;
        }
    }
    deposition_info.init_diagnostics();

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
    ThrusterPlume plume(input.plume);
    plume.find_hits(input, h_scene, h_materials, h_material_ids, hits, hit_positions, num_emit);

    // Copy plume results to GPU
    device_vector<HitInfo> d_hits{hits};
    device_vector<float> d_num_emit{num_emit};

    // Create particle container for carbon atoms and renderer
    ParticleContainer particles{"carbon", max_particles, 1.0f, 1};
    Renderer renderer(input, &h_scene, plume, particles, surfaces);
    renderer.setup(input);

    // generate plume diagnostics if applicable
    if (input.plume.probe) {
        plume.probe();
    }

    // Create timing objects
    size_t step = 0;
    Timer timer;
    cuda::Event start{}, stop_compute{}, stop_copy{};

    if (input.verbosity > 0) std::cout << "Beginning main loop." << std::endl;

    while ((input.display && window.open) || (!input.display && timer.physical_time < input.max_time_s)) {
        // Draw GUI and set up for this frame
        app::begin_frame(step, input, window, renderer, deposition_info, timer);

        // Main computation loop
        if (step > 0 && !app::sim_paused) {
            // Record iteration start time
            start.record();

            // Push particles and sputter from surfaces, then remove those that are out of bounds
            particles.evolve(d_scene, d_materials, d_surface_ids, d_collected,
                             d_hits, d_num_emit, input);

            // record stop time
            stop_compute.record();

            // Track particles collected by each triangle flagged 'collect' and compute diagnostics
            for (int id = 0; id < deposition_info.num_tris; id++) {
                // Copy number of particles collected to CPU
                auto d_begin = d_collected.begin() + deposition_info.global_indices[id];
                thrust::copy(d_begin, d_begin + 1, deposition_info.particles_collected.begin() + id);
            }
            deposition_info.update_diagnostics(input, timer.physical_time);

            // Copy particle data back to CPU
            particles.copy_to_cpu();
            stop_copy.record();

            // timing
            double elapsed_compute = cuda::event_elapsed_time(start, stop_compute);
            double elapsed_copy = cuda::event_elapsed_time(start, stop_copy);
            timer.update_averages(elapsed_compute, elapsed_copy);
        }

        // Draw scene
        renderer.draw(input, app::camera, app::aspect_ratio);

        // Finalize frame and increment timestep
        app::end_frame(input, window);

        // Write output to console and file at regular intervals, plus one additional when simulation terminates
        if ( input.output_interval > 0 && (
            (!app::sim_paused && (step % input.output_interval == 0)) ||
            (!input.display && timer.physical_time >= input.max_time_s) ||
            (input.display && !window.open))) {

            if (input.verbosity > 0) app::write_to_console(step, input, timer);
            deposition_info.write_to_file(step, timer.physical_time);
        }

        if (!app::sim_paused) {
            step++;
            timer.physical_time += input.timestep_s;
        }
    }

    if (input.verbosity > 0) std::cout << "Program terminated successfully." << std::endl;

    return 0;
}
