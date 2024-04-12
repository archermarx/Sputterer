#ifndef _VEC3_H
#define _VEC3_H

#include <iostream>
#include <cmath>

template <typename T>
class Vec3 {
    public:
        T x;
        T y;
        T z;
        Vec3() = default;
        Vec3(T x, T y, T z): x(x), y(y), z(z) {};
        Vec3(T x): x(x), y(x), z(x) {};

        // length and normalization
        inline T lengthSquared() const {
            return x*x + y*y + z*z;
        }

        inline T length() const {
            return std::sqrt(this->lengthSquared());
        }

        void normalize() {
            auto len = this->length();
            x /= len;
            y /= len;
            z /= len;
        }

        inline Vec3<T> cross(Vec3<T> u) const {
            return Vec3<T>(y*u.z - z*u.y, z*u.x - x*u.z, x*u.y - y*u.x);
        }

        // In-place updating functions
        Vec3<T>& operator+=(Vec3<T> u) {
            x += u.x;
            y += u.y;
            z += u.z;
            return *this;
        }

        Vec3<T>& operator-=(Vec3<T> u) {
            x -= u.x;
            y -= u.y;
            z -= u.z;
            return *this;
        }

        Vec3<T>& operator*=(T a) {
            x *= a;
            y *= a;
            z *= a;
            return *this;
        }

        Vec3<T>& operator/=(T a) {
            x /= a;
            y /= a;
            z /= a;
            return *this;
        }

        Vec3<T> operator+(Vec3<T> v) const {
            Vec3<T> vec(x, y, z);
            vec += v;
            return vec;
        }

        Vec3<T> operator-(Vec3<T> v) const {
            Vec3<T> vec(x, y, z);
            vec -= v;
            return vec;
        }

        Vec3<T> operator*(T a) const {
            Vec3<T> vec(x, y, z);
            vec *= a;
            return vec;
        }

        Vec3<T> operator/(T a) const{
            Vec3<T> vec(x, y, z);
            vec /= a;
            return vec;
        }
};

// scalar multiplication (left)
template <typename T>
inline Vec3<T> operator*(T a, Vec3<T> u) {
    return u *= a;
}

template <typename T>
using Point3 = Vec3<T>;

template <typename T>
std::ostream &operator<<(std::ostream &os, Vec3<T> const &v) {
    os << "{" << v.x << ", " << v.y << ", " << v.z << "}";
    return os;
}

#endif