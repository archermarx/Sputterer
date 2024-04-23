#include <iostream>

#include "camera.hpp"

// Returns the view matrix calculated using euler angles and the lookat matrix
glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(center + orientation * distance, center, up);
}

// Returns the projection matrix, given an aspect ratio
glm::mat4 Camera::getProjectionMatrix(float aspectRatio, float min, float max) const {
    return glm::perspective(glm::radians(fov), aspectRatio, min, max);
}

void Camera::processKeyboard(Direction direction, float deltaTime) {
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
    case Direction::Up:
        break;
    case Direction::Down:
        break;
    }
    updateVectors();
}

// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
void Camera::processMouseMovement(float xoffset, float yoffset, CameraMovement movementType) {
    switch (movementType) {
    case CameraMovement::Pan:
        xoffset *= panSensitivity;
        // yoffset *= panSensitivity;
        yaw -= xoffset;
        // pitch -= yoffset;
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
void Camera::processMouseScroll(float yoffset) {
    distance -= zoomSpeed * yoffset;
    if (distance < 0.1f) {
        distance = 0.1f;
    }
    if (distance > MAX_DISTANCE) {
        distance = MAX_DISTANCE;
    }
    updateVectors();
}

// After a change, update camera orientation and position vectors
void Camera::updateVectors(CameraMovement movementType) {
    // make sure that when pitch is out of bounds, screen doesn't get flipped
    pitch = std::min(89.0f, std::max(-89.0f, pitch));

    // Constrain yaw to [0, 360) for to avoid floating point issues at high angles
    yaw = fmod(yaw, 360.0f);

    switch (movementType) {
    case CameraMovement::Orbit: {
        yaw = fmod(yaw, 360.0f);
        // calculates the front vector from the Camera's (updated) Euler Angles
        orientation.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        orientation.y = sin(glm::radians(pitch));
        orientation.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        orientation   = glm::normalize(orientation);

        break;
    }
    case CameraMovement::Pan: {
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