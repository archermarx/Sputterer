#ifndef SPUTTERER_APP_HPP
#define SPUTTERER_APP_HPP

#include "Camera.hpp"
#include "Input.hpp"
#include "ParticleContainer.cuh"
#include "Surface.hpp"
#include "ThrusterPlume.hpp"
#include "Window.hpp"

namespace app {
    // settings
    constexpr unsigned int screen_width = 1360;
    constexpr unsigned int screen_height = 768;
    float aspect_ratio = static_cast<float>(screen_width)/static_cast<float>(screen_height);

    constexpr vec3 carbon_particle_color = {0.05f, 0.05f, 0.05f};
    constexpr float carbon_particle_scale = 0.05;

    Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

    float delta_time = 0.0f;
    float last_x = 0.0;
    float last_y = 0.0;

    bool sim_paused = false;

    void mouse_cursor_callback(GLFWwindow *window, double xpos_in, double ypos_in);
    void pause_callback (GLFWwindow *window, int key, int scancode, int action, int mods);
    void scroll_callback ([[maybe_unused]] GLFWwindow *window, [[maybe_unused]] double xoffset, double yoffset);
    void process_input (GLFWwindow *window);

    class Renderer {
        public:
            BVHRenderer bvh;
            ThrusterPlume &plume;
            ParticleContainer &particles;
            SceneGeometry &geometry;

            Renderer(Input &input, Scene *scene, ThrusterPlume &plume, 
                     ParticleContainer &particles, SceneGeometry &geometry)
                : bvh(scene), plume(plume), particles(particles), geometry(geometry){
                setup(input);
            }

            void setup (Input &input) {
                if (input.display) {
                    geometry.setup_shaders();
                    particles.setup_shaders(carbon_particle_color, carbon_particle_scale);
                    plume.setup_shaders(input.chamber_length_m / 2);
                    bvh.setup_shaders();
                }
            }
            void draw (Input &input) {
                if (input.display) {
                    geometry.draw(camera, aspect_ratio);
                    particles.draw(camera, aspect_ratio);
                    bvh.draw(camera, aspect_ratio);
                    plume.draw(camera, aspect_ratio);
                }
            };
    };

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

    void end_frame(Input &input, Window &window) {
        if (input.display) {
            window.end_render_loop();
            app::process_input(window.window);
        }
    }

    void process_input (GLFWwindow *window) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Forward, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Backward, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Left, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Right, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Up, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            camera.process_keyboard(Direction::Down, delta_time);
        }

        if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
            camera.center = glm::vec3{0.0};
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
