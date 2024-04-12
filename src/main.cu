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

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float aspectRatio = static_cast<float>(SCR_WIDTH) / SCR_HEIGHT;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f;
float lastFrame = 0.0f;

template <typename T>
T readTableEntryAs(toml::table &table, string inputName) {
    auto node = table[inputName];
    bool valid = true;
    T value;

    if constexpr (std::is_same_v<T, string>) {
        if (node.is_string()) {
            value = node.as_string() -> get();
        } else {
            valid = false;
        }
    } else {
        if (node.is_integer()) {
            value = static_cast<T>(node.as_integer() -> get());
        } else if (node.is_boolean()) {
            value = static_cast<T>(node.as_boolean() -> get());
        } else if (node.is_floating_point()) {
            value = static_cast<T>(node.as_floating_point() -> get());
        } else if (node.is_string()) {
            string str = node.as_string() -> get();
            std::istringstream ss(str);
            ss >> value;
        } else {
            valid = false;
        }
    }
    if (!valid) {
        std::cout << "Invalid input for option " << inputName << ".\n Expected value of type " << typeid(T).name() << "\n.";
    }

    return value;
}

vector<Surface> readInput(string filename) {
    std::vector<Surface> surfaces;

    toml::table input;
    try {
        input = toml::parse_file(filename);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return surfaces;
    }

    // Read chamber features
    auto chamber = *input.get_as<toml::table>("chamber");
    auto radius_node = chamber["radius"];
    auto length_node = chamber["length"];

    float radius = readTableEntryAs<float>(chamber, "radius");
    float length = readTableEntryAs<float>(chamber, "length");

    camera.orientation = glm::normalize(glm::vec3(radius, radius, radius));
    camera.distance = 2 * radius;
    camera.yaw = -135;
    camera.pitch = 30;
    camera.updateVectors();

    auto geometry = *input.get_as<toml::array>("geometry");

    for (auto&& elem : geometry) {
        auto tab = elem.as_table();
        string name  = readTableEntryAs<string>(*tab, "name");
        string file  = readTableEntryAs<string>(*tab, "file");
        bool emit    = readTableEntryAs<bool>(*tab, "emit");
        bool collect = readTableEntryAs<bool>(*tab, "collect");
        surfaces.emplace_back(name, file, emit, collect);
    }

    for (auto &surface: surfaces) {
        // enable meshes
        surface.enable();
    }

    return surfaces;
}


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

float lastX = 0.0;
float lastY = 0.0;
bool dragging = false;
bool firstClick = false;

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

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
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

    auto surfaces = readInput(filename);
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
        //  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        for (int i = 0; i < surfaces.size(); i++) {
            surfaces[i].draw(shader);
        }

        window.checkForUpdates();
    }

    return 0;
}
