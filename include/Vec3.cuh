#ifndef _VEC3_H
#define _VEC3_H

template <typename T>
class Vec3 {
    public:
        T x;
        T y;
        T z;
        Vec3(T x, T y, T z): x(x), y(y), z(z) {};
};

template <typename T>
using Point3 = Vec3<T>;

#endif