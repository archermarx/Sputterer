#pragma once
#ifndef SPUTTERER_SHADER_HPP
#define SPUTTERER_SHADER_HPP

#include <string>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "glad/glad.h"

#include "Camera.hpp"

// Declarations
unsigned int create_shader_program (const std::vector<std::string> &sources, const std::vector<unsigned int> &types);

unsigned int compile_shader (const char *source, unsigned int type);

std::string read_from_file (const char *path);

class Shader {
public:
  // Program ID
  unsigned int id;

  Shader () = default;

  void load (const char *vertex_path, const char *fragment_path, const char *geometry_path = nullptr);

  void use () const;

  [[maybe_unused]] void set_bool (const std::string &name, bool value) const;

  [[maybe_unused]] void set_int (const std::string &name, int value) const;

  [[maybe_unused]] void set_float (const std::string &name, float value) const;

  void set_vec3 (const std::string &name, glm::vec3 value) const;

  void set_mat4 (const std::string &name, glm::mat4 value) const;

  [[nodiscard]] GLint get_uniform_location (const std::string &name) const;

  void update_view (const Camera &camera, float aspect_ratio) const;
};

#endif