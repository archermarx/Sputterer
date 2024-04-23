#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>

// Imgui headers
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

#include "window.hpp"

Window::Window(std::string name, unsigned int width, unsigned int height)
    : name(name)
    , width(width)
    , height(height) {

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window and verify that it worked
    this->window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
    if (this->window == NULL) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        open = false;
        return;
    }

    // Make the context of our window the main context on the current thread
    glfwMakeContextCurrent(this->window);

    // Check that GLAD is loaded properly
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        open = false;
        return;
    }

    glEnable(GL_DEPTH_TEST);

    open = true;

    std::cout << "GLFW window initialized." << std::endl;

    // ImGUI initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable keyboard controls

    // Setup platform/renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    std::cout << "ImGUI initialized." << std::endl;
}

void Window::beginRenderLoop() {
    // process user input
    glfwPollEvents();

    // Dear ImGui frame setup
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // draw background
    glClearColor(0.4f, 0.5f, 0.6f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Window::endRenderLoop() {

    // ImGui::Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Swap buffers and determine if we should close
    glfwSwapBuffers(this->window);

    if (glfwWindowShouldClose(this->window)) {
        open = false;
    } else {
        open = true;
    }
}

Window::~Window() {
    // Shut down ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Shut down GLFW
    glfwTerminate();
}
