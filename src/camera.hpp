#pragma once
#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <iosfwd>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "vec3.hpp"

enum class Direction {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
};

enum class CameraMovement { Pan, Orbit };

// Default camera values
constexpr float YAW   = -90.0f;
constexpr float PITCH = 0.0f;

constexpr float PAN_SENSITIVITY   = 0.05f;
constexpr float ORBIT_SENSITIVITY = 0.1f;

constexpr float FOV          = 75.0f;
constexpr float YAW_SPEED    = 100.0f;
constexpr float PITCH_SPEED  = 100.0f;
constexpr float ZOOM_SPEED   = 0.5f;
constexpr float MAX_DISTANCE = 20.0f;

class Camera {
public:
    // camera attributes
    vec3 orientation;
    vec3 front{0.0f, 0.0f, -1.0f};
    vec3 up{0.0f, 1.0f, 0.0f};
    vec3 right{1.0f, 0.0f, 0.0f};
    vec3 worldUp{0.0f, 1.0f, 0.0f};
    vec3 center{0.0f, 0.0f, 0.0f};

    // Euler angles
    float yaw{YAW};
    float pitch{PITCH};
    float distance;

    // camera options
    float pitchSpeed{PITCH_SPEED};
    float yawSpeed{YAW_SPEED};
    float orbitSensitivity{ORBIT_SENSITIVITY};
    float panSensitivity{PAN_SENSITIVITY};
    float zoomSpeed{ZOOM_SPEED};
    float fov{FOV};

    // Returns the view matrix calculated using euler angles and the glfw look at matrix
    [[nodiscard]] glm::mat4 getViewMatrix () const;

    // Returns the projection matrix, given an aspect ratio
    [[nodiscard]] glm::mat4 getProjectionMatrix (float aspectRatio, float min = 0.1f, float max = 100.0f) const;

    // Processes input received from keyboard. Expects a movement direction and a timestep.
    void processKeyboard (Direction direction, float deltaTime);

    // Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void processMouseMovement (float xoffset, float yoffset, CameraMovement movementType);

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void processMouseScroll (float yoffset);

    // After a change, update camera orientation and position vectors
    void updateVectors (CameraMovement movementType = CameraMovement::Orbit);

    // Constructor with vectors
    explicit Camera(vec3 position = {0.0f, 0.0f, 0.0f})
        : orientation(glm::normalize(position))
        , distance(glm::length(position)) {
        updateVectors();
    }
};

#endif