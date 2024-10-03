#ifndef SPUTTERER_WINDOW_H
#define SPUTTERER_WINDOW_H

#include <string>
#include "glad/glad.h"
#include <GLFW/glfw3.h>

#define OPENGL_MAJOR_VERSION 3
#define OPENGL_MINOR_VERSION 3

using std::string;

class Window {
    public:
        string name;
        int width;
        int height;
        bool open;
        GLFWwindow *window;
        bool enabled{false};

        ~Window ();

        void enable ();

        static void begin_render_loop ();

        void initialize_imgui ();

        void end_render_loop ();
};

#endif
