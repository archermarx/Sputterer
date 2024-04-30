#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.hpp"
#include "glad/glad.h"
#include "gl_helpers.hpp"

void shader::load (const char *vertex_path, const char *fragment_path) {
  // 1. Retrieve vertex and fragment source code from file path
  const auto vertexCode = read_from_file(vertex_path);
  const auto fragmentCode = read_from_file(fragment_path);

  // 2. Compile shaders and link
  id = create_shader_program({vertexCode, fragmentCode}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER});
}

void shader::use () const {
  GL_CHECK(glUseProgram(id));
}

GLint shader::get_uniform_location (const std::string &name) const {
  GLint loc;
  GL_CHECK(loc = glGetUniformLocation(id, name.c_str()));
  if (loc < 0) {
    std::cout << "ERROR::SHADER::UNIFORM_NOT_FOUND: " << name << std::endl;
  }
  return loc;
}

void shader::set_bool (const std::string &name, bool value) const {
  GL_CHECK(glUniform1i(get_uniform_location(name), (int) value));
}

void shader::set_int (const std::string &name, int value) const {
  GL_CHECK(glUniform1i(get_uniform_location(name), value));
}

void shader::set_float (const std::string &name, float value) const {
  GL_CHECK(glUniform1f(get_uniform_location(name), value));
}

void shader::set_vec3 (const std::string &name, glm::vec3 value) const {
  GL_CHECK(glUniform3fv(get_uniform_location(name), 1, glm::value_ptr(value)));
}

void shader::set_mat4 (const std::string &name, glm::mat4 value) const {
  GL_CHECK(glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, glm::value_ptr(value)));
}

void shader::update_view (const camera &camera, float aspect_ratio) const {
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
  std::ifstream vFile;

  // ensure ifstream objects can throw exceptsions
  vFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    // open file
    vFile.open(path);
    std::stringstream vStream;
    // read file's buffer contents into streams
    vStream << vFile.rdbuf();
    // close file handlers
    vFile.close();
    // convert stream into string
    contents = vStream.str();
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
  char infoLog[512];
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

  if (!success) {
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
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
    std::cout << "::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  return shader;
}

// Given a list of shader source code and a list of types (e.g. GL_FRAGMENT_SHADER), compile the shaders and link into a
// program
unsigned int create_shader_program (const std::vector<std::string> &sources, const std::vector<unsigned int> &types) {
  unsigned int shaderProgram = glCreateProgram();
  int N = std::min(sources.size(), types.size());

  for (int i = 0; i < N; i++) {
    auto shader = compile_shader(sources[i].c_str(), types[i]);
    glAttachShader(shaderProgram, shader);
    glDeleteShader(shader);
  }

  glLinkProgram(shaderProgram);

  // check for success
  int success;
  char infoLog[512];
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
    std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
  }

  return shaderProgram;
}