#ifndef _TRIANGLE_CUH
#define _TRIANGLE_CUH

#include "Vec3.hpp"
#include <iostream>

std::ostream &operator<< (std::ostream &os, const float3 &v);

inline __host__ __device__ float dot (const float3 a, const float3 b) {
    return {a.x * b.x + a.y * b.y + a.z * b.z};
}

inline __host__ __device__ float3 cross (const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline __host__ __device__ float3 operator+ (const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __host__ __device__ float3 operator- (const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __host__ __device__ float3 operator* (const float a, const float3 b) {
    return {a * b.x, a * b.y, a * b.z};
}

inline __host__ __device__ float3 operator/ (const float3 a, const float b) {
    return {a.x / b, a.y / b, a.z / b};
}

inline __host__ __device__ float3 operator- (const float3 a) {
    return {-a.x, -a.y, -a.z};
}

inline __host__ __device__ float length (const float3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float3 normalize (const float3 v) {
    return v / length(v);
}

inline __host__ __device__ float3 make_float3 (const vec3 &v) {
    return {v.x, v.y, v.z};
}

inline __host__ __device__ float3 make_float3 (const glm::vec4 &v) {
    return {v.x, v.y, v.z};
}

struct Ray {
    float3 origin;
    float3 direction;
};

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 norm;

    __host__ __device__ Triangle (float3 v0, float3 v1, float3 v2)
        : v0(v0)
        , v1(v1)
        , v2(v2) {
        auto e1 = v1 - v0;
        auto e2 = v2 - v0;
        norm    = normalize(cross(e1, e2));
    }
};

struct HitInfo {
    bool   hits{false};
    float  t{0.0};
    float3 norm{0.0};
};

__host__ __device__ HitInfo hits_triangle (Ray ray, Triangle tri);

#endif