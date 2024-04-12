// C++ headers
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include <toml++/toml.hpp>

using std::vector, std::string;

#include "Vec3.hpp"
#include "Surface.hpp"
#include "ParticleContainer.cuh"
#include "Window.hpp"
#include "Shader.hpp"
#include "gl_helpers.hpp"
#include "Camera.hpp"

vector<Surface> readInput(string filename) {

    std::cout << "In readinput" << std::endl;

    std::vector<Surface> surfaces;

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return surfaces;
    }

    auto geometry = *input.get_as<toml::array>("geometry");

    for (auto&& elem : geometry) {
        auto tab = elem.as_table();
        string name = tab->get_as<string>("name")->get();
        string file = tab->get_as<string>("file")->get();
        bool emit = tab->get_as<bool>("emit")->get();
        bool collect = tab->get_as<bool>("collect")->get();
        surfaces.emplace_back(name, file, emit, collect);
    }

    for (auto &surface: surfaces) {
        // enable meshes
        surface.enable();
    }

    return surfaces;
}

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float aspectRatio = static_cast<float>(SCR_WIDTH) / SCR_HEIGHT;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f;
float lastFrame = 0.0f;

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
}

float lastX = 0.0;
float lastY = 0.0;
bool dragging = false;
bool firstClick = false;

void mouseCursorCallback(GLFWwindow *window, double xpos_in, double ypos_in) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
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
    float offsetY = lastY - yPos;
    lastX = xPos;
    lastY = yPos;

    camera.processMouseMovement(offsetX, offsetY);
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

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    auto surfaces = readInput(filename);

    glfwSetCursorPosCallback(window.window, mouseCursorCallback);

    for (const auto& surface: surfaces) {
        std::cout << surface.name << "\n";
        std::cout << surface << "\n";
    }

    glEnable(GL_DEPTH_TEST);

    while (window.open && display) {
        // process user input
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window.window);

        // draw background
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // update camera projection
        shader.setMat4("view", camera.getViewMatrix());
        shader.setMat4("projection", camera.getProjectionMatrix(aspectRatio));

        // Draw geometry
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        for (int i = 0; i < surfaces.size(); i++) {
            surfaces[i].draw(shader);
        }

        window.checkForUpdates();
    }

    return 0;
}
