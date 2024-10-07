#pragma once
#ifndef SPUTTERER_SHADER_H
#define SPUTTERER_SHADER_H

#include <string>
#include <vector>

struct ShaderCode;

struct Shader {
    unsigned int id;
    unsigned int type;
    std::string type_str ();
    Shader(const char *source, unsigned int type);
};

struct ShaderProgram {
    unsigned int id;
    void link (ShaderCode code, const std::string name = "");
    void use () const;

    template <typename T>
    void set_uniform (const std::string &name, T value, bool optional = false) const;
};

#endif
