#ifndef _SHADER_HPP
#define _SHADER_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <glad/glad.h>

// Declarations
unsigned int createShaderProgram(const std::vector<std::string>& sources, const std::vector<unsigned int>& types);
unsigned int compileShader(const char* source, const unsigned int type);
std::string readFromFile(const char* path);

class Shader {
    public:
        // Program ID
        unsigned int ID;

        Shader(const char* vertexPath, const char* fragmentPath);
        void use();
        void setBool(const std::string &name, bool value) const;
        void setInt(const std::string &name, int value) const;
        void setFloat(const std::string &name, float value) const;
};

#endif