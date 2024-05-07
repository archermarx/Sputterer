#pragma once
#ifndef SPUTTERER_TRIANGLE_CUH
#define SPUTTERER_TRIANGLE_CUH

#include <iosfwd>
#include <thrust/host_vector.h>
#include <Shader.hpp>

using thrust::host_vector;

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
  float3 centroid;
  float area;

  __host__ __device__ Triangle (float3 v0, float3 v1, float3 v2)
    : v0(v0), v1(v1), v2(v2), norm{0.0f}, area{-1.0f}, centroid{(v0 + v1 + v2)/3.0f} {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;

    auto cross_prod = cross(e1, e2);
    auto len = length(cross_prod);

    this->area = len/2;
    this->norm = cross_prod/len;
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
  float t{1e30f};
  float3 pos{0.0};
  float3 norm{0.0};
  int id{-1};
};

float3 fminf (float3 a, float3 b);

float3 fmaxf (float3 a, float3 b);

struct BBox {
  float3 lb{1e30f, 1e30f, 1e30f};
  float3 ub{-1e30f, -1e30f, -1e30f};

  void grow (float3 p) { lb = fminf(lb, p), ub = fmaxf(ub, p); }

  float3 extent () { return ub - lb; }

  float area () {
    float3 e = extent();  // box extent
    return e.x*e.y + e.y*e.z + e.z*e.x;
  }

  float3 center () { return 0.5*(lb + ub); }
};

struct BVHNode {
  BBox box;    // 24 bytes, min and maximum extents of axis-aligned bounding box (aabb)
  size_t left_first, tri_count;  // 8 bytes; total: 32 bytes
  __host__ __device__ bool is_leaf () { return tri_count > 0; }
};

struct Scene {
  BVHNode *nodes;
  size_t num_nodes, nodes_used = 0;
  Triangle *triangles;
  size_t *triangle_indices;
  size_t num_tris;
  size_t bvh_depth;

  void build (host_vector<Triangle> &tris, host_vector<size_t> &tri_inds, host_vector<BVHNode> &nodes);

  void build_bvh ();

  void update_node_bounds (size_t node_idx);

  void subdivide_bvh (size_t node_idx, size_t depth);

  float evaluate_sah (size_t node_idx, int axis, float pos);
};

class BVHRenderer {
public:
  Scene *scene;

  static void draw_box (Shader &shader, BBox &box, unsigned int &vao, unsigned int &vbo);

  void draw (Shader &shader, int draw_depth, size_t node_idx = 0);

  void set_buffers ();

private:
  unsigned int vao, vbo;

};

struct Ray {
  float3 origin;
  float3 direction;
  float3 rd;

  __host__ __device__ Ray (float3 orig, float3 dir) : origin(orig), direction(dir), rd({1/dir.x, 1/dir.y, 1/dir.z}) {}

  [[nodiscard]] __host__ __device__ float3 at (float t) const;

  [[nodiscard]] __host__ __device__ HitInfo hits (const Triangle &tri, int id = -1);

  [[nodiscard]] __host__ __device__ HitInfo cast (Scene &scene);

  __host__ __device__ void intersect_tri (const Triangle &triangle, size_t id, HitInfo &closest_hit);

  __host__ __device__ void intersect_bvh (Scene &scene, HitInfo &closest_hit, size_t node_idx = 0);

  [[nodiscard]] __host__ __device__ bool intersect_bbox (const BBox &box, HitInfo &closest_hit);

};

#endif