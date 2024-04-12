#include "Shader.hpp"

// Constructor for Shader type
Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    // 1. Retrieve vertex and fragment source code from file path
    const auto vertexCode = readFromFile(vertexPath);
    const auto fragmentCode = readFromFile(fragmentPath);

    // 2. Compile shaders and link
    ID = createShaderProgram({vertexCode, fragmentCode}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER});
}

void Shader::use() {
    glUseProgram(ID);
}

void Shader::setBool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void Shader::setInt(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}


//----------------------------------------------------------------------------------------------------------------------------------
//                                                 UTILITY FUNCTIONS
//----------------------------------------------------------------------------------------------------------------------------------

// Read the contents of a file into a string
std::string readFromFile(const char* path) {
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
    } catch(std::ifstream::failure const&) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ\n" << path << std::endl;
    }
    return contents;
}

// Compile shader source code into a shader of the provided type, where type is GL_FRAGMENT_SHADER or GL_VERTEX_SHADER or similar.
unsigned int compileShader(const char* source, const unsigned int type) {
    unsigned int shader;
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::";

        switch(type) {
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

// Given a list of shader source code and a list of types (e.g. GL_FRAGMENT_SHADER), compile the shaders and link into a program
unsigned int createShaderProgram(const std::vector<std::string>& sources, const std::vector<unsigned int>& types) {
    unsigned int shaderProgram = glCreateProgram();
    int N = std::min(sources.size(), types.size());

    for (int i = 0; i < N; i++) {
        auto shader = compileShader(sources[i].c_str(), types[i]);
        glAttachShader(shaderProgram, shader);
        glDeleteShader(shader);
    }

    glLinkProgram(shaderProgram);

    // check for success
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
    }

    return shaderProgram;
}