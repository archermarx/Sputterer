#ifndef _SHADER_HPP
#define _SHADER_HPP

#include <string>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "glad/glad.h"

#include "Camera.hpp"

// Declarations
unsigned int createShaderProgram (const std::vector<std::string> &sources, const std::vector<unsigned int> &types);
unsigned int compileShader (const char *source, const unsigned int type);
std::string  readFromFile (const char *path);

class Shader {
public:
    // Program ID
    unsigned int ID;

    Shader(const char *vertexPath, const char *fragmentPath);
    void use ();
    void setBool (const std::string &name, bool value) const;
    void setInt (const std::string &name, int value) const;
    void setFloat (const std::string &name, float value) const;
    void setVec3 (const std::string &name, glm::vec3 value) const;
    void setMat4 (const std::string &name, glm::mat4 value) const;

    GLint getUniformLocation (const std::string &name) const;

    void updateView (const Camera &camera, const float aspectRatio) const;
};

#endif