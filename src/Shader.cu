#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glad/glad.h"

#include "gl_helpers.h"
#include "Camera.h"
#include "Shader.h"
#include "ShaderCode.h"

using std::string;

void Shader::use () const {
    GL_CHECK(glUseProgram(id));
}

void Shader::load (const char *vertex_code, const char *fragment_code, const char *geometry_code) {
    std::vector<string> sources{string(vertex_code), string(fragment_code)};
    std::vector<unsigned int> types {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    if (geometry_code != nullptr) {
        sources.push_back(string(geometry_code));
        types.push_back(GL_GEOMETRY_SHADER);
    }
    id = create_shader_program(sources, types);
}

void Shader::load (ShaderCode code) {
    load(code.vert, code.frag, code.geom);
}

void Shader::update_view (const Camera &camera, float aspect_ratio) const {
    set_uniform("view", camera.get_view_matrix());
    set_uniform("projection", camera.get_projection_matrix(aspect_ratio));
    set_uniform("viewPos", camera.distance*camera.orientation, true);
}

inline void set_uniform_val(GLint loc, bool val)      {GL_CHECK(glUniform1i(loc, (int) val));}
inline void set_uniform_val(GLint loc, int val)       {GL_CHECK(glUniform1i(loc, val));}
inline void set_uniform_val(GLint loc, float val)     {GL_CHECK(glUniform1f(loc, val));}
inline void set_uniform_val(GLint loc, double val)     {GL_CHECK(glUniform1f(loc, val));}
inline void set_uniform_val(GLint loc, glm::vec3 val) {
    GL_CHECK(glUniform3fv(loc, 1, glm::value_ptr(val)));
}
inline void set_uniform_val(GLint loc, glm::mat4 val) {
    GL_CHECK(glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(val)));
}

template <typename T>
void Shader::set_uniform (const std::string &name, T value, bool optional) const {
    // Get uniform location
    GLint loc;
    GL_CHECK(loc = glGetUniformLocation(id, name.c_str()));

    if (loc >= 0) {
        set_uniform_val(loc, value);
    } else if (!optional) {
        std::cerr << "ERROR::SHADER::UNIFORM_NOT_FOUND: " << name << std::endl;
    }
}

template void Shader::set_uniform<bool>(const std::string&, bool, bool) const;
template void Shader::set_uniform<int>(const std::string&, int, bool) const;
template void Shader::set_uniform<float>(const std::string&, float, bool) const;
template void Shader::set_uniform<double>(const std::string&, double, bool) const;
template void Shader::set_uniform<glm::vec3>(const std::string&, glm::vec3, bool) const;
template void Shader::set_uniform<glm::mat4>(const std::string&, glm::mat4, bool) const;

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
