#ifndef _VEC3_HPP
#define _VEC3_HPP

#include <glm/gtc/type_ptr.hpp>
#include <iostream>

using vec3 = glm::vec3;

std::ostream &operator<< (std::ostream &os, const vec3 &v);

#endif