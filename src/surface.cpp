
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include <glad/glad.h>

#include "surface.hpp"
#include "gl_helpers.hpp"

Surface::Surface(std::string name, bool emit, bool collect)
    : name(name)
    , emit(emit)
    , collect(collect) {}
