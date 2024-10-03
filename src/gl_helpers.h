#ifndef SPUTTERER_GL_HELPERS_H
#define SPUTTERER_GL_HELPERS_H

#include <stdio.h>
#include <glad/glad.h>
#include <map>
#include <string>

#define GL_CHECK(code) do {code; gl_check(__FILE__, __LINE__); } while(false)

const static std::map<GLenum, std::string> GL_ERRORS = {
    {0, "GL_NO_ERROR"},
    {0x0500, "GL_INVALID_ENUM"},
    {0x0501, "GL_INVALID_VALUE"},
    {0x0502, "GL_INVALID_OPERATION"},
    {0x0503, "GL_STACK_OVERFLOW"},
    {0x0504, "GL_STACK_UNDERFLOW"},
    {0x0505, "GL_OUT_OF_MEMORY"},
    {0x0506, "GL_INVALID_FRAMEBUFFER_OPERATION"},
    {0x0507, "GL_CONTEXT_LOST"}
};

inline void gl_check(const char *file, int line) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::string errString(GL_ERRORS.at(err));
        fprintf(stderr, "OpenGL Error %x: %s. In file '%s' on line %d\n", err, errString.c_str(), file, line);
        fflush(stderr);
        exit(err);
    }
}


#endif

