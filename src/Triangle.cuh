#pragma once
#ifndef SPUTTERER_TRIANGLE_CUH
#define SPUTTERER_TRIANGLE_CUH

#include <iosfwd>

#include "vec3.hpp"

std::ostream &operator<< (std::ostream &os, const float3 &v);

inline __host__ __device__ float dot (const float3 a, const float3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float3 cross (const float3 a, const float3 b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

inline __host__ __device__ float3 operator+ (const float3 a, const float3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __host__ __device__ float3 operator- (const float3 a, const float3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __host__ __device__ float3 operator* (const float a, const float3 b) {
  return {a*b.x, a*b.y, a*b.z};
}

inline __host__ __device__ float3 operator/ (const float3 a, const float b) {
  return {a.x/b, a.y/b, a.z/b};
}

inline __host__ __device__ float3 operator- (const float3 a) {
  return {-a.x, -a.y, -a.z};
}

inline __host__ __device__ float length (const float3 v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline __host__ __device__ float3 normalize (const float3 v) {
  return v/length(v);
}

inline __host__ __device__ float3 make_float3 (const vec3 &v) {
  return {v.x, v.y, v.z};
}

inline __host__ __device__ float3 make_float3 (const glm::vec4 &v) {
  return {v.x, v.y, v.z};
}

struct Triangle {
  float3 v0;
  float3 v1;
  float3 v2;
  float3 norm;
  float area;

  __host__ __device__ Triangle (float3 v0, float3 v1, float3 v2)
    : v0(v0), v1(v1), v2(v2), norm{0.0f}, area{-1.0f} {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;

    auto cross_prod = cross(e1, e2);
    auto len = length(cross_prod);

    area = len/2;
    norm = cross_prod/len;
  }

  // given random uniform numbers u1 and u2, find a random point on the triangle
  __host__ __device__ float3 sample (float u1, float u2) const {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;

    if (u1 + u2 > 1) {
      // sampled point is outside the triangle and must be transformed back inside
      u1 = 1 - u1;
      u2 = 1 - u2;
    }

    return v0 + (u1*e1 + u2*e2);
  }
};

struct HitInfo {
  bool hits{false};
  float t{0.0};
  float3 norm{0.0};
  int id;
};

struct Ray {
  float3 origin;
  float3 direction;

  [[nodiscard]] __host__ __device__ HitInfo hits (const Triangle &tri, int id = -1) const;

  [[nodiscard]] __host__ __device__ HitInfo cast (const Triangle *tris, size_t num_triangles) const;
};


#endif