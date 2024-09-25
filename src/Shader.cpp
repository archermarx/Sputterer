#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"
#include "glad/glad.h"
#include "gl_helpers.hpp"

using std::string;

void Shader::load (const char *vertex_code, const char *fragment_code, const char *geometry_code) {
    std::vector<string> sources{string(vertex_code), string(fragment_code)};
    std::vector<unsigned int> types {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    if (geometry_code != nullptr) {
        sources.push_back(string(geometry_code));
        types.push_back(GL_GEOMETRY_SHADER);
    }
    id = create_shader_program(sources, types);
}

void Shader::use () const {
    GL_CHECK(glUseProgram(id));
}

GLint Shader::get_uniform_location (const std::string &name) const {
    GLint loc;
    GL_CHECK(loc = glGetUniformLocation(id, name.c_str()));
    if (loc < 0) {
        std::cout << "ERROR::SHADER::UNIFORM_NOT_FOUND: " << name << std::endl;
    }
    return loc;
}

void Shader::set_bool (const std::string &name, bool value) const {
    GL_CHECK(glUniform1i(get_uniform_location(name), (int) value));
}

void Shader::set_int (const std::string &name, int value) const {
    GL_CHECK(glUniform1i(get_uniform_location(name), value));
}

void Shader::set_float (const std::string &name, float value) const {
    auto loc = get_uniform_location(name);
    GL_CHECK(glUniform1f(loc, value));
}

void Shader::set_vec3 (const std::string &name, glm::vec3 value) const {
    GL_CHECK(glUniform3fv(get_uniform_location(name), 1, glm::value_ptr(value)));
}

void Shader::set_mat4 (const std::string &name, glm::mat4 value) const {
    GL_CHECK(glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, glm::value_ptr(value)));
}

void Shader::update_view (const Camera &camera, float aspect_ratio) const {
    set_mat4("view", camera.get_view_matrix());
    set_mat4("projection", camera.get_projection_matrix(aspect_ratio));
    set_vec3("viewPos", camera.distance*camera.orientation);
}

//----------------------------------------------------------------------------------------------------------------------------------
//                                                 UTILITY FUNCTIONS
//----------------------------------------------------------------------------------------------------------------------------------

// Compile shader source code into a shader of the provided type, where type is GL_FRAGMENT_SHADER or GL_VERTEX_SHADER
// or similar.
unsigned int compile_shader (const char *source, const unsigned int type) {
    unsigned int shader;
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    char info_log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cout << "ERROR::SHADER::";

        switch (type) {
        case GL_FRAGMENT_SHADER:
            std::cout << "FRAGMENT";
            break;
        case GL_VERTEX_SHADER:
            std::cout << "VERTEX";
            break;
        default:
            std::cout << "UNKNOWN";
            break;
        }
        std::cout << "::COMPILATION_FAILED\n" << info_log << std::endl;
    }
    return shader;
}

// Given a list of shader source code and a list of types (e.g. GL_FRAGMENT_SHADER), compile the shaders and link into a
// program
unsigned int create_shader_program (const std::vector<std::string> &sources, const std::vector<unsigned int> &types) {
    unsigned int shader_program = glCreateProgram();
    int n = std::min(sources.size(), types.size());

    for (int i = 0; i < n; i++) {
        auto shader = compile_shader(sources[i].c_str(), types[i]);
        glAttachShader(shader_program, shader);
        glDeleteShader(shader);
    }

    glLinkProgram(shader_program);

    // check for success
    int success;
    char info_log[512];
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
        std::cout << "ERROR::SHADER::LINK_FAILED\n" << info_log << std::endl;
    }

    return shader_program;
}
