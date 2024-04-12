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

float deltaTime = 0.0f;
float lastFrame = 0.0f;
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

void processInput(GLFWwindow *window) {
    const float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos += cameraSpeed * cameraFront;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos -= cameraSpeed * cameraFront;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
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

    Window window("Sputterer", 800, 800);

    Shader shader("shaders/shader.vert", "shaders/shader.frag");
    shader.use();

    auto surfaces = readInput(filename);

    for (const auto& surface: surfaces) {
        std::cout << surface.name << "\n";
        std::cout << surface << "\n";
    }

    // Camera info

    const float radius = 10.0f;

    glm::mat4 view = glm::lookAt(
        cameraPos, cameraPos + cameraFront, cameraUp
    );

    // Model matrix
    glm::mat4 model = glm::mat4(1.0f);
    //model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 projection = glm::mat4(1.0f);
    projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    int modelLoc = glGetUniformLocation(shader.ID, "model");
    int viewLoc = glGetUniformLocation(shader.ID, "view");
    int projLoc = glGetUniformLocation(shader.ID, "projection");

    glEnable(GL_DEPTH_TEST);

    while (window.open && display) {


        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window.window);

        view = glm::lookAt(
            cameraPos, cameraPos + cameraFront, cameraUp
        );

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        for (int i = 0; i < surfaces.size(); i++) {
            surfaces[i].draw(shader);
        }

        GL_CHECK( glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model)) );
        GL_CHECK( glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(view)) );
        GL_CHECK( glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection)) );


        window.checkForUpdates();
    }

    return 0;
}
