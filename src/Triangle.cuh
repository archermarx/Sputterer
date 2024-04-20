#pragma once
#include <iostream>

struct Triangle {
    float3 p1;
    float3 p2;
    float3 p3;
};

__global__ void g_translate_triangles (Triangle *t, size_t num_tris) {
    int    i = threadIdx.x + blockIdx.x * blockDim.x;
    float3 v(1.0, 1.0, 1.0);
    if (i < num_tris) {
        t[i].p1 = float3(t[i].p1.x + v.x, t[i].p1.y + v.y, t[i].p1.z + v.z);
        t[i].p2 = float3(t[i].p2.x + v.x, t[i].p2.y + v.y, t[i].p2.z + v.z);
        t[i].p3 = float3(t[i].p3.x + v.x, t[i].p3.y + v.y, t[i].p3.z + v.z);
    }
}