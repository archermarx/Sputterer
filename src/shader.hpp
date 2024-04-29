#pragma once
#ifndef SHADER_HPP
#define SHADER_HPP

#include <string>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "glad/glad.h"

#include "camera.hpp"

// Declarations
unsigned int createShaderProgram (const std::vector<std::string> &sources, const std::vector<unsigned int> &types);
unsigned int compileShader (const char *source, unsigned int type);
std::string  readFromFile (const char *path);

class Shader {
public:
    // Program ID
    unsigned int ID;

    Shader() = default;
    void load (const char *vertexPath, const char *fragmentPath);
    void use () const;

    [[maybe_unused]] void setBool (const std::string &name, bool value) const;
    [[maybe_unused]] void setInt (const std::string &name, int value) const;
    [[maybe_unused]] void setFloat (const std::string &name, float value) const;

    void setVec3 (const std::string &name, glm::vec3 value) const;
    void setMat4 (const std::string &name, glm::mat4 value) const;

    [[nodiscard]] GLint getUniformLocation (const std::string &name) const;

    void updateView (const Camera &camera, float aspectRatio) const;
};

#endif