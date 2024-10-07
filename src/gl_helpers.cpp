#include "gl_helpers.h"

void gl_check (const char *file, int line) {
  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    std::string errString(GL_ERRORS.at(err));
    fprintf(stderr, "OpenGL Error %x: %s. In file '%s' on line %d\n", err, errString.c_str(), file, line);
    fflush(stderr);
    exit(err);
  }
}
