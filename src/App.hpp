#ifndef _APP_H
#define _APP_H

#include "Camera.hpp"

namespace app {
// settings
unsigned int SCR_WIDTH   = 1360;
unsigned int SCR_HEIGHT  = 768;
float        aspectRatio = static_cast<float>(SCR_WIDTH) / SCR_HEIGHT;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f;
float lastFrame = 0.0f;
float lastX     = 0.0;
float lastY     = 0.0;

void processInput (GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Forward, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Backward, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Left, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Right, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Up, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        camera.processKeyboard(Direction::Down, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera.center = glm::vec3{0.0};
    }
}

bool orbiting   = false;
bool panning    = false;
bool firstClick = false;

void mouseCursorCallback (GLFWwindow *window, double xpos_in, double ypos_in) {

    float xPos = static_cast<float>(xpos_in);
    float yPos = static_cast<float>(ypos_in);

    const auto orbitButton = GLFW_MOUSE_BUTTON_RIGHT;
    const auto panButton   = GLFW_MOUSE_BUTTON_MIDDLE;

    // Detection for left mouse drag/release
    if (glfwGetMouseButton(window, orbitButton) == GLFW_RELEASE &&
        glfwGetMouseButton(window, panButton) == GLFW_RELEASE) {
        if (orbiting || panning) {
            orbiting = false;
            panning  = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        return;
    } else if (glfwGetMouseButton(window, panButton) == GLFW_RELEASE) {
        if (panning) {
            panning = false;
        }
        if (!orbiting) {
            orbiting = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            lastX = xPos;
            lastY = yPos;
        }
    } else {
        if (orbiting) {
            orbiting = false;
        }
        if (!panning) {
            panning = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            lastX = xPos;
            lastY = yPos;
        }
    }

    float offsetX = xPos - lastX;
    float offsetY = yPos - lastY;

    lastX = xPos;
    lastY = yPos;

    if (orbiting) {
        camera.processMouseMovement(offsetX, offsetY, CameraMovement::Orbit);
    } else if (panning) {
        camera.processMouseMovement(offsetX, offsetY, CameraMovement::Pan);
    }
}

void scrollCallback (GLFWwindow *window, double xoffset, double yoffset) {
    camera.processMouseScroll(static_cast<float>(yoffset));
}

void framebufferSizeCallback (GLFWwindow *window, int width, int height) {
    aspectRatio = static_cast<float>(width) / height;
    glViewport(0, 0, width, height);
}

} // namespace app

#endif