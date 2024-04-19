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

    // Constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
        : front(glm::vec3(0.0f, 0.0f, -1.0f))
        , orientation(glm::normalize(glm::vec3(posX, posY, posZ)))
        , distance(glm::length(glm::vec3(posX, posY, posZ)))
        , worldUp(upX, upY, upZ)
        , yaw(yaw)
        , pitch(pitch)
        , movementSpeed(SPEED)
        , pitchSpeed(PITCH_SPEED)
        , yawSpeed(YAW_SPEED)
        , zoomSpeed(ZOOM_SPEED)
        , orbitSensitivity(orbitSensitivity)
        , panSensitivity(panSensitivity)
        , fov(FOV) {
        updateVectors();
    }

    // Returns the view matrix calculated using euler angles and the lookat matrix
    glm::mat4 getViewMatrix () {
        return glm::lookAt(center + orientation * distance, center, up);
    }

    // Returns the projection matrix, given an aspect ratio
    glm::mat4 getProjectionMatrix (float aspectRatio, float min = 0.1f, float max = 100.0f) {
        return glm::perspective(glm::radians(fov), aspectRatio, min, max);
    }

    void processKeyboard (Direction direction, float deltaTime) {
        switch (direction) {
        case Direction::Forward:
            pitch += pitchSpeed * deltaTime;
            break;
        case Direction::Backward:
            pitch -= pitchSpeed * deltaTime;
            break;
        case Direction::Left:
            yaw += yawSpeed * deltaTime;
            break;
        case Direction::Right:
            yaw -= yawSpeed * deltaTime;
            break;
        }
        updateVectors();
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void processMouseMovement (float xoffset, float yoffset, CameraMovement movementType) {

        switch (movementType) {
        case CameraMovement::Pan:
            xoffset *= panSensitivity;
            yoffset *= panSensitivity;
            yaw -= xoffset;
            pitch -= yoffset;
            break;
        case CameraMovement::Orbit:
            xoffset *= orbitSensitivity;
            yoffset *= orbitSensitivity;
            yaw += xoffset;
            pitch += yoffset;
            break;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateVectors(movementType);
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void processMouseScroll (float yoffset) {
        distance -= zoomSpeed * yoffset;
        if (distance < 0.1f) {
            distance = 0.1f;
        }
        if (distance > MAX_DISTANCE) {
            distance = MAX_DISTANCE;
        }
        updateVectors();
    }

    void updateVectors (CameraMovement movementType = CameraMovement::Orbit) {
        // make sure that when pitch is out of bounds, screen doesn't get flipped
        pitch = std::min(89.0f, std::max(-89.0f, pitch));

        // Constrain yaw to [0, 360) for to avoid floating point issues at high angles
        yaw = fmod(yaw, 360.0f);

        switch (movementType) {
        case CameraMovement::Orbit: {
            yaw = fmod(yaw, 360.0f);
            std::cout << "Orbiting" << std::endl;
            // calculates the front vector from the Camera's (updated) Euler Angles
            orientation.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            orientation.y = sin(glm::radians(pitch));
            orientation.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            orientation   = glm::normalize(orientation);

            break;
        }
        case CameraMovement::Pan: {
            std::cout << "Panning" << std::endl;
            auto pos = center + distance * orientation;

            orientation.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            orientation.y = sin(glm::radians(pitch));
            orientation.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            orientation   = glm::normalize(orientation);

            center = pos - distance * orientation;

            break;
        }
        }

        front = center - distance * orientation;

        // Also calculate right and up vector
        right = glm::normalize(glm::cross(front, worldUp));
        up    = glm::normalize(glm::cross(right, front));
    }
};

#endif