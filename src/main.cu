// C++ headers
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <type_traits>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::vector, std::string;

#include "Vec3.hpp"
#include "Surface.hpp"
#include "ParticleContainer.cuh"
#include "Window.hpp"
#include "Shader.hpp"
#include "gl_helpers.hpp"
#include "Camera.hpp"
#include "input.hpp"

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float aspectRatio = static_cast<float>(SCR_WIDTH) / SCR_HEIGHT;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f;
float lastFrame = 0.0f;
float lastX = 0.0;
float lastY = 0.0;
bool dragging = false;
bool firstClick = false;

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.processKeyboard(FORWARD, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.processKeyboard(BACKWARD, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.processKeyboard(LEFT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.processKeyboard(RIGHT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        camera.processKeyboard(UP, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        camera.processKeyboard(DOWN, deltaTime);
    }
}

void mouseCursorCallback(GLFWwindow *window, double xpos_in, double ypos_in) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
        if (dragging) {
            dragging = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        return;
    }

    float xPos = static_cast<float>(xpos_in);
    float yPos = static_cast<float>(ypos_in);

    if (!dragging) {
        dragging = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        lastX = xPos;
        lastY = yPos;
    }

    float offsetX = xPos - lastX;
    float offsetY = yPos - lastY;
    lastX = xPos;
    lastY = yPos;

    camera.processMouseMovement(offsetX, offsetY);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(static_cast<float>(yoffset));
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    aspectRatio = static_cast<float>(width) / height;
    glViewport(0, 0, width, height);
}

int main(int argc, char * argv[]) {
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

    Window window("Sputterer", SCR_WIDTH, SCR_HEIGHT);

    glfwSetFramebufferSizeCallback(window.window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window.window, mouseCursorCallback);
    glfwSetScrollCallback(window.window, scrollCallback);

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    Input input(filename);
    input.read();

    camera.orientation = glm::normalize(glm::vec3(input.chamberRadius));
    camera.distance = 2 * input.chamberRadius;
    camera.yaw = -135;
    camera.pitch = 30;
    camera.updateVectors();

    // for (const auto& surface: input.surfaces) {
    //     std::cout << surface.name << "\n";
    //     std::cout << surface << "\n";
    // }

    glEnable(GL_DEPTH_TEST);

    ParticleContainer pc{"noname", 1.0f, 1};


    std::cout << "Particle x: " << input.particle_x.size() << std::endl;;

    pc.addParticles(
        input.particle_x, input.particle_y, input.particle_z,
        input.particle_vx, input.particle_vy, input.particle_vz,
        input.particle_w
    );


    glm::vec3 particleColor{0.0f, 0.2f, 0.8f};
    glm::vec3 particleScale{0.1f};
    glm::vec3 particleTranslate{0.0f};

    Surface particleMesh("Particle", "o_sphere.obj", false, false, particleScale, particleTranslate, particleColor);
    particleMesh.enable();

    while (window.open && display) {
        // process user input
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window.window);

        // draw background
        glClearColor(0.4f, 0.5f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // update camera projection
        shader.setMat4("view", camera.getViewMatrix());
        shader.setMat4("projection", camera.getProjectionMatrix(aspectRatio));
        shader.setVec3("viewPos", camera.distance * camera.orientation);

        glm::mat4 model;

        for (const auto& surface: input.surfaces) {
            // set the model matrix
            model = glm::translate(glm::mat4(1.0f), surface.translate);
            model = glm::scale(model, surface.scale);
            shader.setMat4("model", model);
            shader.setVec3("objectColor", surface.color);
            surface.draw(shader);
        }

        for (int i = 0; i < pc.numParticles; i++) {
            auto pos = glm::vec3{pc.position_x.at(i), pc.position_y.at(i), pc.position_z.at(i)};
            model = glm::translate(glm::mat4(1.0f), pos);
            model = glm::scale(model, particleMesh.scale);
            shader.setMat4("model", model);
            shader.setVec3("objectColor", particleMesh.color);
            particleMesh.draw(shader);
        }

        window.checkForUpdates();
    }

    return 0;
}
