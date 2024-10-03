#ifndef SPUTTERER_APP_HPP
#define SPUTTERER_APP_HPP

#include "Camera.hpp"
#include "DepositionInfo.hpp"
#include "Input.hpp"
#include "ParticleContainer.cuh"
#include "Surface.hpp"
#include "ThrusterPlume.hpp"
#include "Timer.hpp"
#include "Window.hpp"
#include "Renderer.hpp"

namespace app {
    // settings
    constexpr unsigned int screen_width = 1360;
    constexpr unsigned int screen_height = 768;
    float aspect_ratio = static_cast<float>(screen_width)/static_cast<float>(screen_height);

    constexpr auto imgui_flags = ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_AlwaysAutoResize |
                                 ImGuiWindowFlags_NoSavedSettings;

    Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

    // wall clock times
    float delta_time = 0.0f;
    auto current_time = std::chrono::system_clock::now();
    auto last_time = std::chrono::system_clock::now();

    float last_x = 0.0;
    float last_y = 0.0;

    bool sim_paused = false;

    void mouse_cursor_callback(GLFWwindow *window, double xpos_in, double ypos_in);
    void pause_callback (GLFWwindow *window, int key, int scancode, int action, int mods);
    void scroll_callback ([[maybe_unused]] GLFWwindow *window, [[maybe_unused]] double xoffset, double yoffset);
    void process_input (GLFWwindow *window);

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

    void draw_deposition_panel(size_t step, Input &input, Renderer &renderer, DepositionInfo &dep_info, Timer timer) {
        using namespace ImGui;
        auto table_flags = ImGuiTableFlags_BordersH;
        ImVec2 bottom_left = ImVec2(0, GetIO().DisplaySize.y);
        SetNextWindowPos(bottom_left, ImGuiCond_Always, ImVec2(0.0, 1.0));
        Begin("Particle collection info", nullptr, app::imgui_flags);
        if (BeginTable("Table", 4, table_flags)) {
            TableNextRow();
            TableNextColumn();
            Text("Surface name");
            TableNextColumn();
            Text("Triangle ID");
            TableNextColumn();
            Text("Particles collected");
            TableNextColumn();
            Text("Deposition rate [um/kh]");
            for (int tri = 0; tri < dep_info.num_tris; tri++) {
                TableNextRow();
                TableNextColumn();
                Text("%s", dep_info.surface_names[tri].c_str());
                TableNextColumn();
                Text("%li", dep_info.local_indices[tri]);
                TableNextColumn();
                Text("%d", dep_info.particles_collected[tri]);
                TableNextColumn();
                Text("%.3f", dep_info.deposition_rates[tri]);
            }
            EndTable();
        }
        End();
    }

    void draw_settings_panel(size_t step, Input &input, Renderer &renderer, Timer timer) {
        using namespace ImGui;

        auto &bvh = renderer.bvh;
        auto &plume = renderer.plume;
        auto &particles = renderer.particles;
        ImVec2 top_right = ImVec2(GetIO().DisplaySize.x, 0);
        SetNextWindowPos(top_right, ImGuiCond_Always, ImVec2(1.0, 0.0));
        Begin("Display settings", nullptr, imgui_flags);
        if (BeginTable("split", 1)) {
            TableNextColumn();
            Checkbox("Show grid", &renderer.grid.enabled);
            TableNextColumn();
            Text("Grid opacity");
            TableNextColumn();
            SliderFloat("##grid_opacity", &renderer.grid.opacity, 0.0, 1.0);
            TableNextColumn();
            Checkbox("Show plume cone", &plume.render);
            TableNextColumn();
            Checkbox("Show plume particles", &plume.particles.render);
            TableNextColumn();
            Text("Plume particle scale");
            TableNextColumn();
            SliderFloat("##plume_particle_scale", &plume.particles.scale, 0, 0.3);
            TableNextColumn();
            Checkbox("Show sputtered particles", &particles.render);
            TableNextColumn();
            Text("Sputtered particle scale");
            TableNextColumn();
            SliderFloat("##sputtered_particle_scale", &particles.scale, 0, 0.3);
            TableNextColumn();
            Checkbox("Show bounding boxes", &bvh.enabled);
            TableNextColumn();
            Text("Bounding box depth");
            TableNextColumn();
            SliderInt("##bvh_depth", &bvh.draw_depth, 0, bvh.draw_depth);
            EndTable();
        }
        End();
    }

    void draw_timing_panel(size_t step, Input &input, Renderer &renderer, Timer timer) {
        using namespace ImGui;

        ImVec2 bottom_right = ImVec2(GetIO().DisplaySize.x, GetIO().DisplaySize.y);
        SetNextWindowPos(bottom_right, ImGuiCond_Always, ImVec2(1.0, 1.0));
        Begin("Timing info", nullptr);

        auto framerate_ms = 1000.0 / timer.dt_smoothed;
        auto transfer_percentage = (1.0 - timer.avg_time_compute/timer.avg_time_total) * 100.0;
        auto compute_percentage = (timer.avg_time_total/timer.dt_smoothed)*100;

        Text(
            "Simulation step %li (%s)\n"
            "Simulation time: %s\n"
            "Compute time: %.3f ms (%.2f%% data transfer)\n"
            "Frame time: %.3f ms (%.1f fps, %.2f%% compute\n"
            "Particles: %i",
            step,
            print_time(input.timestep_s).c_str(),
            print_time(timer.physical_time).c_str(),
            timer.avg_time_compute,
            transfer_percentage,
            timer.dt_smoothed,
            framerate_ms,
            compute_percentage,
            renderer.particles.num_particles
        );
        End();
    }

    void draw_gui(size_t step, Input &input, Renderer &renderer, DepositionInfo &dep_info, Timer &timer) {
        draw_timing_panel(step, input, renderer, timer);
        draw_settings_panel(step, input, renderer, timer);
        draw_deposition_panel(step, input, renderer, dep_info, timer);
    }

    void begin_frame(size_t step, Input &input, Window &window, Renderer &renderer, DepositionInfo &dep_info, Timer &timer) {
        if (!input.display) return;
        window.begin_render_loop();
        draw_gui(step, input, renderer, dep_info, timer);
        
        // Record iteration timing information
        using namespace std::chrono;
        current_time = system_clock::now();
        auto diff = current_time - last_time;
        last_time = current_time;
        delta_time = static_cast<float>(duration_cast<microseconds>(diff).count())/1e6;
        timer.dt_smoothed = exp_avg(timer.dt_smoothed, delta_time * 1000, time_const);
    }

    void end_frame (Input &input, Window &window) {
        if (!input.display) return;
        window.end_render_loop();
        app::process_input(window.window);
    }

    void write_to_console (size_t step, Input &input, Timer &timer) {
        std::cout << "  Step " << step
                  << ", Simulation time: " << print_time(timer.physical_time)
                  << ", Timestep: " << print_time(input.timestep_s)
                  << ", Avg. step time: " << timer.dt_smoothed << " ms\n";
    }

    Window initialize(Input &input) {
        Window window{.name = "Sputterer", .width = screen_width, .height = screen_height};
        if (input.display) {
            window.enable();
            camera.initialize(input.chamber_radius_m);
            glfwSetKeyCallback(window.window, pause_callback);
            glfwSetCursorPosCallback(window.window, mouse_cursor_callback);
            glfwSetScrollCallback(window.window, scroll_callback);
            window.initialize_imgui();
            sim_paused = true;
        }
        return window;
    }

    void process_input (GLFWwindow *window) {
        bool alt_pressed = false;
        if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
            // re-enable after other camera weirdness fixed
            // alt_pressed = true;
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            if (alt_pressed) {
                camera.process_keyboard(Direction::MoveForward, delta_time);
            } else {
                camera.process_keyboard(Direction::OrbitForward, delta_time);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            if (alt_pressed) {
                camera.process_keyboard(Direction::MoveBackward, delta_time);
            } else {
                camera.process_keyboard(Direction::OrbitBackward, delta_time);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            if (alt_pressed) {
                camera.process_keyboard(Direction::MoveLeft, delta_time);
            } else {
                camera.process_keyboard(Direction::OrbitLeft, delta_time);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            if (alt_pressed) {
                camera.process_keyboard(Direction::MoveRight, delta_time);
            } else {
                camera.process_keyboard(Direction::OrbitRight, delta_time);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            camera.process_keyboard(Direction::MoveUp, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            camera.process_keyboard(Direction::MoveDown, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
            camera.center = glm::vec3{0.0};
        }

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camera.process_keyboard(Direction::ZoomIn, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera.process_keyboard(Direction::ZoomOut, delta_time);
        }
    }

    void pause_callback (GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                sim_paused = !sim_paused;
            }
        }
    }

    bool orbiting = false;
    bool panning = false;

    void mouse_cursor_callback (GLFWwindow *window, double xpos_in, double ypos_in) {

        auto io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            return;
        }

        auto x_pos = static_cast<float>(xpos_in);
        auto y_pos = static_cast<float>(ypos_in);

        const auto orbit_button = GLFW_MOUSE_BUTTON_RIGHT;
        const auto pan_button = GLFW_MOUSE_BUTTON_MIDDLE;

        // Detection for left mouse drag/release
        if (glfwGetMouseButton(window, orbit_button) == GLFW_RELEASE &&
                glfwGetMouseButton(window, pan_button) == GLFW_RELEASE) {
            if (orbiting || panning) {
                orbiting = false;
                panning = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            return;
        } else if (glfwGetMouseButton(window, pan_button) == GLFW_RELEASE) {
            if (panning) {
                panning = false;
            }
            if (!orbiting) {
                orbiting = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                last_x = x_pos;
                last_y = y_pos;
            }
        } else {
            if (orbiting) {
                orbiting = false;
            }
            if (!panning) {
                panning = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                last_x = x_pos;
                last_y = y_pos;
            }
        }

        float offset_x = x_pos - last_x;
        float offset_y = y_pos - last_y;

        last_x = x_pos;
        last_y = y_pos;

        if (orbiting) {
            camera.process_mouse_movement(offset_x, offset_y, CameraMovement::Orbit);
        } else if (panning) {
            camera.process_mouse_movement(offset_x, offset_y, CameraMovement::Pan);
        }
    }

    void scroll_callback ([[maybe_unused]] GLFWwindow *window, [[maybe_unused]] double xoffset, double yoffset) {
        camera.process_mouse_scroll(static_cast<float>(yoffset));
    }
} // namespace app

#endif
