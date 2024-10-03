#pragma once
#ifndef SPUTTERER_SHADER_HPP
#define SPUTTERER_SHADER_HPP

#include <string>
#include <vector>

#include "Camera.hpp"

unsigned int create_shader_program (const std::vector<std::string> &sources, const std::vector<unsigned int> &types);
unsigned int compile_shader (const char *source, unsigned int type);

struct ShaderCode;

class Shader {
    public:
        // Program ID
        unsigned int id;

        Shader () = default;

        void load (ShaderCode shader);
        void load (const char *vertex_path, const char *fragment_path, const char *geometry_path = nullptr);
        void use () const;

        void update_view (const Camera &camera, float aspect_ratio) const;

        template <typename T>
        void set_uniform (const std::string &name, T value, bool optional = false) const;
};


#endif
