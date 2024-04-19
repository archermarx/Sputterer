
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include <glad/glad.h>

#include "Surface.hpp"
#include "gl_helpers.hpp"

Surface::Surface(std::string name, bool emit, bool collect, glm::vec3 scale, glm::vec3 translate, glm::vec3 color)
    : name(name)
    , emit(emit)
    , collect(collect)
    , mesh()
    , scale(scale)
    , translate(translate)
    , color(color) {}