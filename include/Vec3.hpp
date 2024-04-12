#ifndef _VEC3_H
#define _VEC3_H

#include <iostream>

template <typename T>
class Vec3 {
    public:
        T x;
        T y;
        T z;
        Vec3() = default;
        Vec3(T x, T y, T z): x(x), y(y), z(z) {};
};

template <typename T>
using Point3 = Vec3<T>;

template <typename T>
std::ostream &operator<<(std::ostream &os, Vec3<T> const &v) {
    os << "{" << v.x << ", " << v.y << ", " << v.z << "}";
    return os;
}

#endif