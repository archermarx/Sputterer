#ifndef _WINDOW_HPP
#define _WINDOW_HPP

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <string>

#define OPENGL_MAJOR_VERSION 3
#define OPENGL_MINOR_VERSION 3

using std::string;

class Window {
    public:
        std::string name;
        int width;
        int height;
        bool open;
        GLFWwindow* window;

        Window(string name, int width, int height);
        ~Window();
        void checkForUpdates();

};

#endif