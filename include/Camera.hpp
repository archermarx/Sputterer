#ifndef _CAMERA_HPP
#define _CAMERA_HPP

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
const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  45.0f;

class Camera {
    public:
        // camera attributes
        glm::vec3 position;
        glm::vec3 front;
        glm::vec3 up;
        glm::vec3 right;
        glm::vec3 worldUp;

        // Euler angles
        float yaw;
        float pitch;

        // camera options
        float movementSpeed;
        float mouseSensitivity;
        float zoom;

        // Constructor with vectors
        Camera( glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
                float yaw = YAW,
                float pitch = PITCH)
                : front(glm::vec3(0.0f, 0.0f, -1.0f))
                , position(position)
                , worldUp(up)
                , yaw(yaw)
                , pitch(pitch)
                , movementSpeed(SPEED)
                , mouseSensitivity(SENSITIVITY)
                , zoom(ZOOM)
        {
            updateVectors();
        }

        // Constructor with scalar values
        Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
            : front(glm::vec3(0.0f, 0.0f, -1.0f))
            , position(posX, posY, posZ)
            , worldUp(upX, upY, upZ)
            , yaw(yaw)
            , pitch(pitch)
            , movementSpeed(SPEED)
            , mouseSensitivity(SENSITIVITY)
            , zoom(ZOOM)
        {
            updateVectors();
        }

        // Returns the view matrix calculated using euler angles and the lookat matrix
        glm::mat4 getViewMatrix() {
            return glm::lookAt(position, position + front, up);
        }

        // Returns the projection matrix, given an aspect ratio
        glm::mat4 getProjectionMatrix(float aspectRatio, float min = 0.1f, float max = 100.0f) {
            return glm::perspective(glm::radians(zoom), aspectRatio, min, max);
        }

        void processKeyboard(CameraMovement direction, float deltaTime) {
            float velocity = movementSpeed * deltaTime;
            switch (direction) {
                case FORWARD:
                    position += front * velocity;
                    break;
                case BACKWARD:
                    position -= front * velocity;
                    break;
                case LEFT:
                    position -= right * velocity;
                    break;
                case RIGHT:
                    position += right * velocity;
                    break;
                case UP:
                    position += up * velocity;
                    break;
                case DOWN:
                    position -= up * velocity;
                    break;
            }
        }

        // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
        void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {
            xoffset *= mouseSensitivity;
            yoffset *= mouseSensitivity;

            yaw   += xoffset;
            pitch += yoffset;

            // make sure that when pitch is out of bounds, screen doesn't get flipped
            if (constrainPitch)
            {
                if (pitch > 89.0f)
                    pitch = 89.0f;
                if (pitch < -89.0f)
                    pitch = -89.0f;
            }

            // update Front, Right and Up Vectors using the updated Euler angles
            updateVectors();
        }

        // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
        void processMouseScroll(float yoffset) {
            zoom -= yoffset;
            if (zoom < 1.0f)
                zoom = 1.0f;
            if (zoom > 45.0f)
                zoom = 45.0f;
        }

    private:
        void updateVectors() {
            // calculates the front vector from the Camera's (updated) Euler Angles
            glm::vec3 newFront;
            newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            newFront.y = sin(glm::radians(pitch));
            newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            front = glm::normalize(newFront);

            // Also calculate right and up vector
            right = glm::normalize(glm::cross(front, worldUp));
            up    = glm::normalize(glm::cross(right, front));

        }

};

#endif