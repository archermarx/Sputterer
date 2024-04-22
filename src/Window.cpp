#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.hpp"

Window::Window(std::string name, int width, int height)
    : name(name)
    , width(width)
    , height(height) {

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window and verify that it worked
    window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        open = false;
        return;
    }

    // Make the context of our window the main context on the current thread
    glfwMakeContextCurrent(window);

    // Check that GLAD is loaded properly
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        open = false;
        return;
    }

    open = true;
}

void Window::checkForUpdates() {
    // Check and call events, then swap buffers
    glfwSwapBuffers(window);

    if (glfwWindowShouldClose(window)) {
        open = false;
    } else {
        open = true;
    }
}

Window::~Window() {
    glfwTerminate();
}
