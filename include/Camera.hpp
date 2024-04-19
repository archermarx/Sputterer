#ifndef _CAMERA_HPP
#define _CAMERA_HPP

#include <iostream>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Window.hpp"

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN,
};

// Default camera values
constexpr float YAW         = -90.0f;
constexpr float PITCH       =  0.0f;
constexpr float SPEED       =  2.5f;
constexpr float SENSITIVITY =  0.1f;
constexpr float FOV         = 75.0f;
constexpr float YAW_SPEED   = 100.0f;
constexpr float PITCH_SPEED = 100.0f;
constexpr float ZOOM_SPEED  = 0.5f;
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
        float mouseSensitivity;
        float zoomSpeed;
        float fov;

        // Constructor with vectors
        Camera( glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f),
                float yaw = YAW,
                float pitch = PITCH)
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
                , mouseSensitivity(SENSITIVITY)
                , fov(FOV)
        {
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
            , mouseSensitivity(SENSITIVITY)
            , fov(FOV)
        {
            updateVectors();
        }

        // Returns the view matrix calculated using euler angles and the lookat matrix
        glm::mat4 getViewMatrix() {
            return glm::lookAt(orientation * distance, center, up);
        }

        // Returns the projection matrix, given an aspect ratio
        glm::mat4 getProjectionMatrix(float aspectRatio, float min = 0.1f, float max = 100.0f) {
            return glm::perspective(glm::radians(fov), aspectRatio, min, max);
        }

        void processKeyboard(CameraMovement direction, float deltaTime) {
            switch (direction) {
                case FORWARD:
                    pitch += pitchSpeed * deltaTime;
                    break;
                case BACKWARD:
                    pitch -= pitchSpeed * deltaTime;
                    break;
                case LEFT:
                    //position -= right * velocity;
                    yaw += yawSpeed * deltaTime;
                    break;
                case RIGHT:
                    yaw -= yawSpeed * deltaTime;
                    //position += right * velocity;
                    break;
                // case UP:
                //     position += up * velocity;
                //     break;
                // case DOWN:
                //     position -= up * velocity;
                //     break;
            }
            updateVectors();
        }

        // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
        void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {
            xoffset *= mouseSensitivity;
            yoffset *= mouseSensitivity;

            yaw   += xoffset;
            pitch += yoffset;

            // update Front, Right and Up Vectors using the updated Euler angles
            updateVectors();
        }

        // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
        void processMouseScroll(float yoffset) {
            distance -= zoomSpeed * yoffset;
            if (distance < 0.1f) {
                distance = 0.1f;
            }
            if (distance > MAX_DISTANCE) {
                distance = MAX_DISTANCE;
            }
            updateVectors();
        }

        void updateVectors() {
            // make sure that when pitch is out of bounds, screen doesn't get flipped
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;

            // Constrain yaw to [0, 360) for to avoid floating point issues at high angles
            yaw = fmod(yaw, 360.0f);

            // calculates the front vector from the Camera's (updated) Euler Angles
            orientation.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            orientation.y = sin(glm::radians(pitch));
            orientation.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            front = center - distance * orientation;

            // Also calculate right and up vector
            right = glm::normalize(glm::cross(front, worldUp));
            up    = glm::normalize(glm::cross(right, front));

        }
};

#endif