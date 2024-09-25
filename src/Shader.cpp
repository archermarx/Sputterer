#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"
#include "glad/glad.h"
#include "gl_helpers.hpp"

std::string fixed_path(const char *file) {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));

    return dir_path + "/../shaders/" + std::string(file);
}

void Shader::load (const char *vertex_path, const char *fragment_path, const char *geometry_path) {
    // TODO: shaders should probably actually just be strings in source code so they're embedded in executable
    // 1. Retrieve vertex and fragment source code from file path
    const auto vertex_code = read_from_file(fixed_path(vertex_path).c_str());
    const auto fragment_code = read_from_file(fixed_path(fragment_path).c_str());

    // 2. Compile shaders and link
    if (geometry_path == nullptr) {
        id = create_shader_program({vertex_code, fragment_code}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER});
    } else {
        const auto geometry_code = read_from_file(fixed_path(geometry_path).c_str());
        id = create_shader_program(
                {vertex_code, fragment_code, geometry_code}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER}
                );
    }
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

// Read the contents of a file into a string
std::string read_from_file (const char *path) {
    std::string contents;
    std::ifstream v_file;

    // ensure ifstream objects can throw exceptsions
    v_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        // open file
        v_file.open(path);
        std::stringstream v_stream;
        // read file's buffer contents into streams
        v_stream << v_file.rdbuf();
        // close file handlers
        v_file.close();
        // convert stream into string
        contents = v_stream.str();
    } catch (std::ifstream::failure const &) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ\n" << path << std::endl;
    }
    return contents;
}

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
