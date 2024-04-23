#pragma once
#ifndef VEC3_HPP
#define VEC3_HPP

#include <glm/gtc/type_ptr.hpp>
#include <iosfwd>

using vec3 = glm::vec3;

std::ostream &operator<< (std::ostream &os, const vec3 &v);

#endif