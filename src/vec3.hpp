#pragma once
#ifndef SPUTTERER_VEC3_HPP
#define SPUTTERER_VEC3_HPP

#include <glm/gtc/type_ptr.hpp>
#include <iosfwd>

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

std::ostream &operator<< (std::ostream &os, const vec3 &v);

#endif