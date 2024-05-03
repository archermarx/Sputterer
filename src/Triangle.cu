#include <iostream>
#include "Triangle.cuh"

std::ostream &operator<< (std::ostream &os, const float3 &v) {
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os;
}

#define MIN_T 100'000
#define TOL 1e-6

__host__ __device__ HitInfo Ray::hits (const Triangle &tri, int id) const {
  HitInfo info;

  // Find vectors for two edges sharing v1
  auto edge1 = tri.v1 - tri.v0;
  auto edge2 = tri.v2 - tri.v0;

  // Begin calculating determinant
  auto pvec = cross(this->direction, edge2);
  auto det = dot(edge1, pvec);

  // If determinant is near zero, ray lies in plane of triangle
  if (abs(det) < TOL) {
    return info;
  }

  // Calculate distance from v0 to ray origin
  auto tvec = this->origin - tri.v0;

  // Calculate u parameter and test bounds
  auto u = dot(tvec, pvec)/det;
  if (u < 0.0 || u > 1.0) {
    return info;
  }

  auto qvec = cross(tvec, edge1);

  // Calculate v parameter and test bounds
  auto v = dot(this->direction, qvec)/det;
  if (v < 0.0 || u + v > 1.0) {
    return info;
  }
  // Calculate t, ray intersects triangle
  auto t = dot(edge2, qvec)/det;

  info.hits = true;
  info.t = t;
  info.id = id;

  // Orient direction properly
  if (dot(this->direction, tri.norm) > 0) {
    info.norm = -tri.norm;
  } else {
    info.norm = tri.norm;
  }

  return info;
}

__host__ __device__ HitInfo Ray::cast (const Triangle *tris, size_t num_triangles) const {
  HitInfo closest_hit{.hits = false, .t = static_cast<float>(MIN_T), .norm = {0.0, 0.0, 0.0}, .id = -1};
  for (int i = 0; i < num_triangles; i++) {
    auto current_hit = this->hits(tris[i], i);
    if (current_hit.hits && current_hit.t < closest_hit.t && current_hit.t >= 0) {
      closest_hit = current_hit;
    }
  }
  return closest_hit;
}