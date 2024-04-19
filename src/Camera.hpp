#ifndef _CAMERA_HPP
#define _CAMERA_HPP

#include <iostream>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Window.hpp"

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
constexpr float SPEED = 2.5f;

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
    glm::vec3 orientation;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    glm::vec3 center;

    // Euler angles
    float yaw;
    float pitch;
    float distance;

    // camera options
    float movementSpeed;
    float pitchSpeed;
    float yawSpeed;
    float orbitSensitivity;
    float panSensitivity;
    float zoomSpeed;
    float fov;

    // Returns the view matrix calculated using euler angles and the lookat matrix
    glm::mat4 getViewMatrix ();

    // Returns the projection matrix, given an aspect ratio
    glm::mat4 getProjectionMatrix (float aspectRatio, float min = 0.1f, float max = 100.0f);

    // Processes input recieved from keyboard. Expects a movement direction and a timestep.
    void processKeyboard (Direction direction, float deltaTime);

    // Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void processMouseMovement (float xoffset, float yoffset, CameraMovement movementType);

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void processMouseScroll (float yoffset);

    // After a change, update camera orientation and position vectors
    void updateVectors (CameraMovement movementType = CameraMovement::Orbit);

    // Constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f), float yaw = YAW, float pitch = PITCH)
        : front(glm::vec3(0.0f, 0.0f, -1.0f))
        , orientation(glm::normalize(position))
        , distance(glm::length(position))
        , worldUp(up)
        , yaw(yaw)
        , pitch(pitch)
        , movementSpeed(SPEED)
        , pitchSpeed(PITCH_SPEED)
        , yawSpeed(YAW_SPEED)
        , zoomSpeed(ZOOM_SPEED)
        , orbitSensitivity(ORBIT_SENSITIVITY)
        , panSensitivity(PAN_SENSITIVITY)
        , fov(FOV) {
        updateVectors();
    }
};

#endif