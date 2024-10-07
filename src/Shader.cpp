#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glad/glad.h"

#include "gl_helpers.h"
#include "Shader.h"
#include "ShaderCode.h"

constexpr size_t shader_log_size = 512;

std::string Shader::type_str() {
    switch (this->type) {
    case (GL_VERTEX_SHADER):
        return "vertex";
    case (GL_FRAGMENT_SHADER):
        return "fragment";
    case (GL_GEOMETRY_SHADER):
        return "geometry";
    default:
        return "unknown";
    }
}

Shader::Shader(const char *source, unsigned int type) {
    // Compile shader
    this->id = glCreateShader(type);
    this->type = type;
    glShaderSource(this->id, 1, &source, nullptr);
    glCompileShader(this->id);

    // Check for success
    GLint success = GL_FALSE;
    GLchar info_log[shader_log_size] = {'\0'};
    glGetShaderiv(this->id, GL_COMPILE_STATUS, &success);

    if (success == GL_FALSE) {
        std::cout << "ERROR: " << this->type_str() << " shader compilation failed" << std::endl;
        glGetShaderInfoLog(this->id, shader_log_size, NULL, info_log);
        std::cout << "INFO: " << info_log << std::endl;
    } else {
        std::cout << this->type_str() << " shader compiled successfully" << std::endl;
    }
}

void ShaderProgram::link(ShaderCode code, const std::string name) {
    this->id = glCreateProgram();

    // Compile shaders and link program
    if (code.vert != nullptr) {
        Shader vert(code.vert, GL_VERTEX_SHADER);
        glAttachShader(this->id, vert.id);
    }

    if (code.frag != nullptr) {
        Shader frag(code.frag, GL_FRAGMENT_SHADER);
        glAttachShader(this->id, frag.id);
    }

    if (code.geom != nullptr) {
        Shader geom(code.geom, GL_GEOMETRY_SHADER);
        glAttachShader(this->id, geom.id);
    }

    glLinkProgram(this->id);

    // Check for success
    GLint success = GL_FALSE;
    GLchar info_log[shader_log_size] = {'\0'};
    glGetProgramiv(this->id, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
        std::cout << "ERROR: shader program " << name << " linking failed" << std::endl;
        glGetProgramInfoLog(this->id, shader_log_size, nullptr, info_log);
        std::cout << "INFO: " << info_log << std::endl;
    } else {
        std::cout << "shader program " << name << " linked successfully" << std::endl;
    }
}

void ShaderProgram::use () const {
    GL_CHECK(glUseProgram(this->id));
}

inline void set_uniform_val (GLint loc, bool val) {
    GL_CHECK(glUniform1i(loc, (int)val));
}
inline void set_uniform_val (GLint loc, int val) {
    GL_CHECK(glUniform1i(loc, val));
}
inline void set_uniform_val (GLint loc, float val) {
    GL_CHECK(glUniform1f(loc, val));
}
inline void set_uniform_val (GLint loc, double val) {
    GL_CHECK(glUniform1f(loc, static_cast<float>(val)));
}
inline void set_uniform_val (GLint loc, glm::vec3 val) {
    GL_CHECK(glUniform3fv(loc, 1, glm::value_ptr(val)));
}
inline void set_uniform_val (GLint loc, glm::mat4 val) {
    GL_CHECK(glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
}

template <typename T>
void ShaderProgram::set_uniform(const std::string &name, T value, bool optional) const {
    // Get uniform location
    GLint loc;
    GL_CHECK(loc = glGetUniformLocation(id, name.c_str()));

    if (loc >= 0) {
        set_uniform_val(loc, value);
    } else if (!optional) {
        std::cerr << "ERROR::SHADER::UNIFORM_NOT_FOUND: " << name << std::endl;
    }
}

template void ShaderProgram::set_uniform<bool>(const std::string &, bool, bool) const;
template void ShaderProgram::set_uniform<int>(const std::string &, int, bool) const;
template void ShaderProgram::set_uniform<float>(const std::string &, float, bool) const;
template void ShaderProgram::set_uniform<double>(const std::string &, double, bool) const;
template void ShaderProgram::set_uniform<glm::vec3>(const std::string &, glm::vec3, bool) const;
template void ShaderProgram::set_uniform<glm::mat4>(const std::string &, glm::mat4, bool) const;