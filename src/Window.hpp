#pragma once
#ifndef SPUTTERER_WINDOW_HPP
#define SPUTTERER_WINDOW_HPP

#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <string>

#define OPENGL_MAJOR_VERSION 3
#define OPENGL_MINOR_VERSION 3

using std::string;

class Window {
public:
  string name;
  unsigned int width;
  unsigned int height;
  bool open;
  GLFWwindow *window;
  bool enabled{false};

  ~Window ();

  void enable ();

  static void begin_render_loop ();

  void end_render_loop ();
};

#endif