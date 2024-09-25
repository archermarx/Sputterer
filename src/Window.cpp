#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>

// ImGUI headers
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

#include "Window.hpp"

void Window::enable () {

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create window and verify that it worked
  this->window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), name.c_str(), nullptr
                                  , nullptr);
  if (this->window == nullptr) {
    std::cerr << "Failed to create GLFW window\n";
    glfwTerminate();
    open = false;
    return;
  }

  // Make the context of our window the main context on the current thread
  glfwMakeContextCurrent(this->window);
  glfwSwapInterval(1); // enable vsync

  // Check that GLAD is loaded properly
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    this->open = false;
    return;
  }

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  this->open = true;

  std::cout << "GLFW window initialized." << std::endl;

  this->enabled = true;

  initialize_imgui();
}


void Window::initialize_imgui () {
  // ImGUI initialization
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable keyboard controls

  // Setup platform/renderer backends
  ImGui_ImplGlfw_InitForOpenGL(this->window, true);
  ImGui_ImplOpenGL3_Init();
  std::cout << "ImGUI initialized." << std::endl;
}


void Window::begin_render_loop () {
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

void Window::end_render_loop () {

  // ImGui::Rendering
  ImGui::Render();

  // Swap buffers and determine if we should close
  glfwGetFramebufferSize(this->window, &width, &height);
  glViewport(0, 0, this->width, this->height);

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glfwSwapBuffers(this->window);

  if (glfwWindowShouldClose(this->window)) {
    this->open = false;
  } else {
    this->open = true;
  }
}

Window::~Window () {
  if (this->enabled) {
    // Shut down ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Shut down GLFW
    glfwDestroyWindow(this->window);
    glfwTerminate();
  }
}
