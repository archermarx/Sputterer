#include <iostream>
#include "Triangle.cuh"

std::ostream &operator<< (std::ostream &os, const float3 &v) {
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os;
}

__host__ __device__ HitInfo Ray::hits (const Triangle &tri, int id) {
  HitInfo info;

  // Find vectors for two edges sharing v1
  auto edge1 = tri.v1 - tri.v0;
  auto edge2 = tri.v2 - tri.v0;

  // Begin calculating determinant
  auto pvec = cross(this->direction, edge2);
  auto det = dot(edge1, pvec);

  // If determinant is near zero, ray lies in plane of triangle
  if (abs(det) < 1e-6) {
    return info;
  }

  // Calculate distance from v0 to ray origin
  auto tvec = this->origin - tri.v0;

  // Calculate u parameter and test bounds
  auto inv_det = 1.0f/det;
  auto u = dot(tvec, pvec)*inv_det;
  if (u < 0.0 || u > 1.0) {
    return info;
  }

  auto qvec = cross(tvec, edge1);

  // Calculate v parameter and test bounds
  auto v = dot(this->direction, qvec)*inv_det;
  if (v < 0.0 || u + v > 1.0) {
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
  subdivide_bvh(root_node_idx);
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
  node.lb = float3(1e30f);
  node.ub = float3(-1e30f);

  if (!check_index(num_tris, node.left_first + node.tri_count - 1, "triangle")) return;

  for (size_t first = node.left_first, i = 0; i < node.tri_count; i++) {
    auto &leaf_tri_idx = triangle_indices[first + i];
    auto &leaf_tri = triangles[leaf_tri_idx];
    node.lb = fminf(node.lb, leaf_tri.v0);
    node.lb = fminf(node.lb, leaf_tri.v1);
    node.lb = fminf(node.lb, leaf_tri.v2);
    node.ub = fmaxf(node.ub, leaf_tri.v0);
    node.ub = fmaxf(node.ub, leaf_tri.v1);
    node.ub = fmaxf(node.ub, leaf_tri.v2);
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

void Scene::subdivide_bvh (size_t node_idx) {

  if (!check_index(num_nodes, node_idx, "BVH node")) return;

  // don't split nodes with two or fewer triangles
  auto &node = nodes[node_idx];

  // split bounding box along longest axis
  if (node.tri_count <= 2) return;
  auto extent = node.ub - node.lb;
  int axis = 0;
  if (extent.y > at(extent, axis)) {
    axis = 1;
  }
  if (extent.z > at(extent, axis)) {
    axis = 2;
  }

  float split_pos = at(node.lb, axis) + at(extent, axis)*0.5f;

  // split group in two halves in place
  int i = node.left_first;
  int j = i + node.tri_count - 1;
  while (i <= j) {
    if (at(triangles[triangle_indices[i]].centroid, axis) < split_pos) {
      i++;
    } else {
      auto tmp = triangle_indices[i];
      triangle_indices[i] = triangle_indices[j];
      triangle_indices[j] = tmp;
      j--;
    }
  }

  // create child nodes for each halves
  int left_count = i - node.left_first;

  // check that our partition is real (i.e. that we have at least one triangle on each side)
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
  subdivide_bvh(left_child_idx);
  subdivide_bvh(right_child_idx);
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

__host__ __device__ bool Ray::intersect_bbox (const float3 lb, const float3 ub, HitInfo &closest_hit) {
  float dx = 1.0/direction.x;
  float tx1 = (lb.x - origin.x)*dx;
  float tx2 = (ub.x - origin.x)*dx;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float dy = 1.0/direction.y;
  float ty1 = (lb.y - origin.y)*dy;
  float ty2 = (ub.y - origin.y)*dy;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float dz = 1.0/direction.z;
  float tz1 = (lb.z - origin.z)*dz;
  float tz2 = (ub.z - origin.z)*dz;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  return tmax >= tmin && tmin < closest_hit.t && tmax > 0;
}

__host__ __device__ void Ray::intersect_bvh (Scene &scene, HitInfo &closest_hit, size_t node_idx) {

  auto &node = scene.nodes[node_idx];
  if (!intersect_bbox(node.lb, node.ub, closest_hit)) return;

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

