#include <iostream>
#include "Triangle.cuh"

#include "gl_helpers.hpp"

std::ostream &operator<< (std::ostream &os, const float3 &v) {
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os;
}

__host__ __device__ HitInfo Ray::hits (const Triangle &tri, int id) {
  HitInfo info;

  // expand triangle by a small amount
  auto v0 = tri.v0;
  auto v1 = tri.v1;
  auto v2 = tri.v2;

  // Find vectors for two edges sharing v1
  auto edge1 = v1 - v0;
  auto edge2 = v2 - v0;

  // Begin calculating determinant
  auto pvec = cross(this->direction, edge2);
  auto det = dot(edge1, pvec);

  // If determinant is near zero, ray lies in plane of triangle
  if (abs(det) < 1e-8f) {
    return info;
  }

  // Calculate distance from v0 to ray origin
  auto tvec = this->origin - tri.v0;

  // Calculate u parameter and test bounds
  auto inv_det = 1.0f/det;
  auto u = dot(tvec, pvec)*inv_det;
  if (u < 0.0f || u > 1.0f) {
    return info;
  }

  auto qvec = cross(tvec, edge1);

  // Calculate v parameter and test bounds
  auto v = dot(this->direction, qvec)*inv_det;
  if (v < 0.0f || u + v > 1.0f) {
    return info;
  }
  // Calculate t, ray intersects triangle
  auto t = dot(edge2, qvec)*inv_det;

  info.hits = true;
  info.t = t;
  info.id = id;
  info.pos = this->at(t);

  // Orient direction properly
  if (dot(this->direction, tri.norm) > 0) {
    info.norm = -tri.norm;
  } else {
    info.norm = tri.norm;
  }

  return info;
}

__host__ __device__ float3 Ray::at (float t) const {
  return this->origin + t*this->direction;
}

void Scene::build (host_vector<Triangle> &h_tris, host_vector<size_t> &h_tri_inds, host_vector<BVHNode> &h_nodes) {
  this->triangles = h_tris.data();
  this->num_tris = h_tris.size();

  h_tri_inds.resize(this->num_tris);
  this->triangle_indices = h_tri_inds.data();

  this->num_nodes = 2*this->num_tris + 1;
  h_nodes.resize(this->num_nodes);
  this->nodes = h_nodes.data();
  this->nodes_used = 0;

  this->build_bvh();
}

void Scene::build_bvh () {
  // populate triangle indices
  for (int i = 0; i < num_tris; i++) {
    triangle_indices[i] = i;
  }

  // assign all triangles to root node
  size_t root_node_idx = 0;
  nodes_used = 1;
  auto &root = nodes[0];
  root.left_first = 0;
  root.tri_count = num_tris;

  // get scene bounding box
  update_node_bounds(root_node_idx);

  // recursively subdivide
  this->bvh_depth = 0;
  subdivide_bvh(root_node_idx, 0);
}

float3 fminf (float3 a, float3 b) {
  return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

float3 fmaxf (float3 a, float3 b) {
  return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

bool check_index (size_t num_nodes, size_t node_idx, std::string type) {
  if (node_idx >= num_nodes) {
    std::cerr << "OUT OF BOUNDS:: " << type << " index " << node_idx << " exceeds maximum number index (" << node_idx
              << ")!\n";
    return false;
  }
  return true;
}

void Scene::update_node_bounds (size_t node_idx) {

  if (!check_index(num_nodes, node_idx, "BVH node")) return;

  auto &node = nodes[node_idx];
  auto &[lb, ub] = node.box;

  if (!check_index(num_tris, node.left_first + node.tri_count - 1, "triangle")) return;

  for (size_t first = node.left_first, i = 0; i < node.tri_count; i++) {
    auto &leaf_tri_idx = triangle_indices[first + i];
    auto &leaf_tri = triangles[leaf_tri_idx];
    lb = fminf(lb, leaf_tri.v0);
    lb = fminf(lb, leaf_tri.v1);
    lb = fminf(lb, leaf_tri.v2);
    ub = fmaxf(ub, leaf_tri.v0);
    ub = fmaxf(ub, leaf_tri.v1);
    ub = fmaxf(ub, leaf_tri.v2);
  }
}

float at (float3 v, size_t i) {
  switch (i) {
    case (0): {
      return v.x;
    }
    case (1) : {
      return v.y;
    }
    case (2): {
      return v.z;
    }
    default: {
      return NAN;
    }
  }
}

float Scene::evaluate_sah (size_t node_idx, int axis, float pos) {
  auto &node = this->nodes[node_idx];
  BBox left_box, right_box;
  int left_count = 0, right_count = 0;
  for (size_t i = 0; i < node.tri_count; i++) {
    Triangle &triangle = this->triangles[this->triangle_indices[node.left_first + i]];
    if (at(triangle.centroid, axis) < pos) {
      left_count++;
      left_box.grow(triangle.v0);
      left_box.grow(triangle.v1);
      left_box.grow(triangle.v2);
    } else {
      right_count++;
      right_box.grow(triangle.v0);
      right_box.grow(triangle.v1);
      right_box.grow(triangle.v2);
    }
  }
  float cost = left_count*left_box.area() + right_count*right_box.area();

  return cost > 0 ? cost : 1e30f;
}

void Scene::subdivide_bvh (size_t node_idx, size_t depth) {

  if (!check_index(num_nodes, node_idx, "BVH node")) return;

  // don't split nodes with two or fewer triangles
  auto &node = nodes[node_idx];

  // calculate parent cost
  float parent_cost = node.tri_count*node.box.area();
  auto extent = node.box.extent();

  // determine bounding box split using surface area heuristic

  int num_positions = 12;
  float delta = 1.0/num_positions;

  int best_axis = -1;
  float best_pos = 0, best_cost = 1e30f;
  for (int ax = 0; ax < 3; ax++) {
    for (size_t i = 1; i < num_positions - 1; i++) {

      auto candidate_pos = at(node.box.lb, ax) + i*delta*at(extent, ax);

      float cost = evaluate_sah(node_idx, ax, candidate_pos);
      if (cost < best_cost) {
        best_pos = candidate_pos, best_axis = ax, best_cost = cost;
      }
    }
  }
  if (best_cost >= parent_cost) return;

  // split group in two halves in place
  int i = node.left_first;
  int j = i + node.tri_count - 1;
  while (i <= j) {
    auto idx = triangle_indices[i];
    if (at(triangles[idx].centroid, best_axis) < best_pos) {
      i++;
    } else {
      triangle_indices[i] = triangle_indices[j];
      triangle_indices[j] = idx;
      j--;
    }
  }

  // abort split if one side is empty
  int left_count = i - node.left_first;
  if (left_count == 0 || left_count == node.tri_count) return;

  // create child nodes and assign triangles/primitives to each
  size_t left_child_idx = nodes_used;
  size_t right_child_idx = left_child_idx + 1;
  nodes_used += 2;

  nodes[left_child_idx].left_first = node.left_first;
  nodes[left_child_idx].tri_count = left_count;
  nodes[right_child_idx].left_first = i;
  nodes[right_child_idx].tri_count = node.tri_count - left_count;
  // since tri_count is now zero, left_first is now the left child index
  node.left_first = left_child_idx;
  node.tri_count = 0;

  // update child node bounds
  update_node_bounds(left_child_idx);
  update_node_bounds(right_child_idx);

  // recurse
  this->bvh_depth = std::max(this->bvh_depth, depth);
  subdivide_bvh(left_child_idx, depth + 1);
  subdivide_bvh(right_child_idx, depth + 1);
}


__host__ __device__ HitInfo Ray::cast (Scene &scene) {
  HitInfo closest_hit{};

#if 0
  for (int i = 0; i < scene.num_tris; i++) {
    intersect_tri(scene.triangles[i], i, closest_hit);
  }
#else
  // bounding volume heirarchy intersection, starting at root node
  intersect_bvh(scene, closest_hit, 0);
#endif
  return closest_hit;
}

__host__ __device__ void Ray::intersect_tri (const Triangle &triangle, size_t id, HitInfo &closest_hit) {
  auto hit = this->hits(triangle, id);
  if (hit.hits && hit.t < closest_hit.t && hit.t >= 0) {
    closest_hit = hit;
  }
}

__host__ __device__ bool Ray::intersect_bbox (const BBox &box, HitInfo &closest_hit) {
  const auto &[lb, ub] = box;
  float tx1 = (lb.x - origin.x)*rd.x;
  float tx2 = (ub.x - origin.x)*rd.x;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float ty1 = (lb.y - origin.y)*rd.y;
  float ty2 = (ub.y - origin.y)*rd.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (lb.z - origin.z)*rd.z;
  float tz2 = (ub.z - origin.z)*rd.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  return tmax >= tmin && tmin < closest_hit.t && tmax > 0;
}

__host__ __device__ void Ray::intersect_bvh (Scene &scene, HitInfo &closest_hit, size_t node_idx) {
  auto &node = scene.nodes[node_idx];
  if (!intersect_bbox(node.box, closest_hit)) return;

  if (node.is_leaf()) {
    for (size_t i = 0; i < node.tri_count; i++) {
      size_t tri_idx = scene.triangle_indices[node.left_first + i];
      intersect_tri(scene.triangles[tri_idx], tri_idx, closest_hit);
    }
  } else {
    intersect_bvh(scene, closest_hit, node.left_first);
    intersect_bvh(scene, closest_hit, node.left_first + 1);
  }
}

void BVHRenderer::set_buffers () {
  glGenBuffers(1, &vbo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  float points[] = {0.0, 0.0, 0.0};
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(points), 0);
  glBindVertexArray(0);
}

void BVHRenderer::draw_box (Shader &shader, BBox &box, unsigned int &vao, unsigned int &vbo) {
  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  auto center = box.center();
  auto extent = box.ub - box.lb;
  float points[] = {center.x, center.y, center.z};
  shader.set_vec3("extent", {extent.x, extent.y, extent.z});
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_DYNAMIC_DRAW);
  glDrawArrays(GL_POINTS, 0, 1);
}

void BVHRenderer::draw (Shader &shader, int draw_depth, size_t node_idx) {

  if (draw_depth == 0) {
    return;
  }

  auto &node = this->scene->nodes[node_idx];
  // draw current box
  draw_box(shader, node.box, this->vao, this->vbo);

  if (node.is_leaf()) {
    return;
  } else {
    // recursively draw children
    draw(shader, draw_depth - 1, node.left_first);
    draw(shader, draw_depth - 1, node.left_first + 1);
  }
}
