#pragma once
#ifndef WINDOW_HPP
#define WINDOW_HPP

#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <string>

#define OPENGL_MAJOR_VERSION 3
#define OPENGL_MINOR_VERSION 3

using std::string;

class Window {
public:
    string       name;
    unsigned int width;
    unsigned int height;
    bool         open;
    GLFWwindow  *window;

    Window(string name, unsigned int width, unsigned int height);
    ~Window();
    static void beginRenderLoop ();
    void        endRenderLoop ();
};

#endif