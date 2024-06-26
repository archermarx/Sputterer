#include <iostream>
#include "vec3.hpp"

std::ostream &operator<< (std::ostream &os, const vec3 &v) {
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os;
}