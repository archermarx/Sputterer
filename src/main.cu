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

ParticleContainer find_plume_hits (Input &input,
                                   Scene &h_scene,
                                   host_vector<Material> &h_materials,
                                   host_vector<size_t> &h_material_ids,
                                   host_vector<HitInfo> &hits,
                                   host_vector<float3> &hit_positions,
                                   host_vector<float> &num_emit) {
    using namespace constants;
    host_vector <float3> vel;
    host_vector<float> ws;

    // plume coordinate system
    auto plume = input.plume;
    auto plume_up = vec3{0.0, 1.0, 0.0};
    auto plume_right = cross(plume.direction, plume_up);
    plume_up = cross(plume_right, plume.direction);

    auto incident = constants::xenon;
    auto target = constants::carbon;

    auto [main_fraction, scattered_fraction, _] = plume.current_fractions();
    auto beam_fraction = main_fraction + scattered_fraction;
    main_fraction = main_fraction/beam_fraction;

    // tune the number of rays to get a max emission probability close to 1
    float target_prob = 0.95;
    float tol = 0.1;
    float max_iters = 5;
    float max_emit_prob = 0.0;
    int num_rays = 10'000;

    for (int iter = 0; iter < max_iters; iter++) {
        // TODO: do this on GPU
        // TODO: use low-discrepancy sampler for this
        float max_emit = 0.0;
        for (int i = 0; i < num_rays; i++) {
            // select whether ray comes from main beam or scattered beam based on
            // fraction of beam that is scattered vs main
            auto u = rand_uniform();
            double div_angle, beam_energy;
            if (u < main_fraction) {
                div_angle = plume.main_divergence_angle();
                beam_energy = plume.beam_energy_eV;
            } else {
                div_angle = plume.scattered_divergence_angle();
                beam_energy = plume.scattered_energy_eV;
            }

            auto azimuth = rand_uniform(0, 2*constants::pi);
            auto elevation = abs(rand_normal(0, div_angle/sqrt(2.0)));

            auto direction = cos(elevation)*plume.direction + sin(elevation)*(cos(azimuth)*plume_right + sin(azimuth)*plume_up);
            Ray r{make_float3(plume.origin + direction*1e-3f), normalize(make_float3(direction))};

            auto hit = r.cast(h_scene);

            if (hit.hits) {
                // get material that we hit
                auto &mat = h_materials[h_material_ids[hit.id]];

                // Only record hit if material is allowed to sputter
                if (mat.sputter) {
                    hits.push_back(hit);
                    auto hit_pos = r.at(hit.t);
                    hit_positions.push_back(hit_pos);
                    vel.push_back({0.0f, 0.0f, 0.0f});
                    ws.push_back(0.0f);

                    auto cos_hit_angle = static_cast<double>(dot(r.direction, -hit.norm));
                    auto hit_angle = acos(cos_hit_angle);

                    auto yield = sputtering_yield(beam_energy, hit_angle, incident, target);
                    auto n_emit = yield*plume.beam_current_A*beam_fraction/q_e/num_rays/input.particle_weight;
                    if (n_emit > max_emit)
                        max_emit = n_emit;

                    num_emit.push_back(n_emit);
                }
            }
        }
        // basic proportional controller
        max_emit_prob = max_emit*input.timestep_s;
        num_rays = static_cast<int>(num_rays*max_emit_prob/target_prob);
        if (abs(max_emit_prob - target_prob) < tol) {
            break;
        }
    }

    if (input.verbosity > 0) {
        std::cout << "Max emission probability: " << max_emit_prob << std::endl;
        std::cout << "Number of plume rays: " << num_rays << std::endl;
    }

    ParticleContainer pc_plume{"plume", hit_positions.size()};
    pc_plume.add_particles(hit_positions, vel, ws);
    return pc_plume;
}

int main (int argc, char *argv[]) {
    using namespace constants;
    // Initialize GPU
    device_vector<int> init{0};

    std::string filename = "input.toml";
    if (argc > 1) filename = argv[1];

    Input input = read_input(filename);

    // construct triangles
    // TODO: move to a function, maybe within input
    host_vector<Triangle> h_triangles;
    host_vector<size_t> h_material_ids;
    host_vector<Material> h_materials;

    host_vector<char> h_to_collect;
    std::vector<int> collect_inds_global;
    std::vector<int> collect_inds_local;

    std::vector<string> surface_names;

    for (size_t id = 0; id < input.surfaces.size(); id++) {
        const auto &surf = input.surfaces.at(id);
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
    auto pc_plume = find_plume_hits(input, h_scene, h_materials, h_material_ids, hits, hit_positions, num_emit);

    // Copy plume results to GPU
    device_vector<HitInfo> d_hits{hits};
    device_vector<float> d_num_emit{num_emit};

    // Create particle container for carbon atoms
    ParticleContainer pc{"carbon", max_particles, 1.0f, 1};

    // Display objects
    Window window{.name = "Sputterer", .width = app::screen_width, .height = app::screen_height};
    Shader mesh_shader{}, particle_shader{}, bvh_shader{};
    BVHRenderer bvh(&h_scene);
    app::camera.initialize(input.chamber_radius_m);

    if (input.display) {
        // enable window
        window.enable();

        // Register window callbacks
        glfwSetKeyCallback(window.window, app::pause_callback);
        glfwSetCursorPosCallback(window.window, app::mouse_cursor_callback);
        glfwSetScrollCallback(window.window, app::scroll_callback);

        window.initialize_imgui();

        // Load mesh shader
        mesh_shader.load("shader.vert", "shader.frag");

        // initialize mesh buffers
        for (auto &surf: input.surfaces) {
            surf.mesh.set_buffers();
        }

        // Load particle shader
        particle_shader.load("particle.vert", "particle.frag");
        particle_shader.use();
        constexpr vec3 particle_scale{0.01f};
        particle_shader.set_vec3("scale", particle_scale);
        particle_shader.set_vec3("cameraRight", app::camera.right);
        particle_shader.set_vec3("cameraUp", app::camera.up);

        // TODO: have geometric primitives stored as strings in a c++ source file
        // Set up particle meshes
        pc.mesh.read_from_obj("../o_rect.obj");
        pc.set_buffers();

        pc_plume.mesh.read_from_obj("../o_rect.obj");
        pc_plume.set_buffers();

        // Load plume shader
        input.plume.setup_shaders(input.chamber_length_m / 2);

        // set up BVH rendering
        bvh_shader.load("bvh.vert", "bvh.frag", "bvh.geom");
        bvh_shader.use();
        bvh.set_buffers();
    }

    // Create timing objects
    size_t frame = 0;

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

    bool render_plume_particles = true;
    bool render_sputtered_particles = true;
    bool render_bvh = false;
    int bvh_draw_depth = h_scene.bvh_depth;

    // Pause simulation if displaying
    app::simulation_paused = input.display;

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
                    "transfer)   \nFrame time: %.3f ms (%.1f fps, %.2f%% compute)   \nParticles: %i", frame, print_time(
                        input.timestep_s).c_str(), print_time(physical_time).c_str(), avg_time_compute,
                    (1.0f - avg_time_compute/avg_time_total)*100, delta_time_smoothed, 1000/delta_time_smoothed,
                    (avg_time_total/delta_time_smoothed)*100, pc.num_particles);
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

            flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;

            ImVec2 top_right = ImVec2(ImGui::GetIO().DisplaySize.x - padding, 0);
            ImGui::SetNextWindowPos(top_right, ImGuiCond_Always, ImVec2(1.0, 0.0));
            ImGui::Begin("Options", nullptr, flags);
            if (ImGui::BeginTable("split", 1)) {
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show plume cone", &input.plume.render_cone);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show plume particles", &render_plume_particles);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show sputtered particles", &render_sputtered_particles);
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show bounding boxes", &render_bvh);
                ImGui::TableNextColumn();
                ImGui::SliderInt("BVH depth  ", &bvh_draw_depth, 0, h_scene.bvh_depth);
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
        if (!app::simulation_paused) {
            physical_time += input.timestep_s;
        }
        delta_time_smoothed = (1 - time_const)*delta_time_smoothed + time_const*app::delta_time*1000;

        // Main computations
        if (frame > 0 && !app::simulation_paused) {
            start.record();

            // Push particles and sputter from surfaces
            pc.evolve(d_scene, d_materials, d_surface_ids, d_collected,
                      d_hits, d_num_emit, input.particle_weight,
                      input.timestep_s);

            // flag particles that are out of bounds
            pc.flag_out_of_bounds(input.chamber_radius_m, input.chamber_length_m);

            // remove particles with negative weight (out of bounds and phantom emitted particles)
            pc.remove_flagged_particles();

            // record stop time
            stop_compute.record();

            // Track particles collected by each triangle flagged 'collect'
            for (int id = 0; id < collect_inds_global.size(); id++) {
                auto d_begin = d_collected.begin() + collect_inds_global[id];
                thrust::copy(d_begin, d_begin + 1, collected.begin() + id);
            }

            // Copy particle data back to CPU
            pc.copy_to_cpu();
            stop_copy.record();

            // timing
            float elapsed_compute, elapsed_copy;
            elapsed_compute = cuda::event_elapsed_time(start, stop_compute);
            elapsed_copy = cuda::event_elapsed_time(start, stop_copy);

            avg_time_compute = (1 - time_const)*avg_time_compute + time_const*elapsed_compute;
            avg_time_total = (1 - time_const)*avg_time_total + time_const*elapsed_copy;
        }

        // Rendering
        if (input.display) {

            // get camera matrix for use in particle and plume shaders
            auto cam = app::camera.get_projection_matrix(app::aspect_ratio)*app::camera.get_view_matrix();

            // 1. draw user-provided geometry

            // update update camera uniforms
            mesh_shader.use();
            mesh_shader.update_view(app::camera, app::aspect_ratio);

            for (const auto &surface: input.surfaces) {
                // set the model matrix and object color per surface
                mesh_shader.use();
                mesh_shader.set_mat4("model", surface.transform.get_matrix());
                mesh_shader.set_vec3("objectColor", surface.color);
                surface.mesh.draw();
            }

            // 2. draw particles (instanced!)
            if (render_sputtered_particles && pc.num_particles > 0) {
                // activate particle shader
                particle_shader.use();
                particle_shader.set_vec3("cameraRight", app::camera.right);
                particle_shader.set_vec3("cameraUp", app::camera.up);
                particle_shader.set_vec3("objectColor", {0.05f, 0.05f, 0.05f});
                particle_shader.set_mat4("camera", cam);

                // draw particles
                pc.draw();
            }

            // Draw bounding volume heirarchy
            if (render_bvh) {
                bvh_shader.use();
                bvh_shader.set_mat4("camera", cam);
                bvh.draw(bvh_shader, bvh_draw_depth);
            }

            // Draw plume particles
            if (render_plume_particles) {
                particle_shader.use();
                particle_shader.set_vec3("cameraUp", app::camera.up);
                particle_shader.set_vec3("objectColor", {0.2f, 0.75f, 0.94f});
                particle_shader.set_mat4("camera", cam);
                pc_plume.draw();
            }

            // Draw translucent plume cone
            input.plume.draw(cam);
        }

        if (!app::simulation_paused && physical_time > next_output_time ||
                (!input.display && physical_time >= input.max_time_s) ||
                (input.display && !window.open)) {
            // Write output to console at regular intervals, plus one additional when simulation terminates
            std::cout << "Step " << frame << ", Simulation time: " << print_time(physical_time)
                << ", Timestep: " << print_time(input.timestep_s) << ", Avg. step time: " << delta_time_smoothed
                << " ms" << std::endl;

            // write output to file
            next_output_time += input.output_interval_s;}

        if (input.display) {
            window.end_render_loop();
            app::process_input(window.window);
        }

        if (!app::simulation_paused) frame += 1;
    }

    if (input.verbosity > 0) {
        std::cout << "Program terminated successfully." << std::endl;
    }

    return 0;
}
