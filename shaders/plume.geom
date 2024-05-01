#version 330 core

#define MAX_VERTICES 102
#define PI 3.1415926538

layout (points) in;
layout (triangle_strip, max_vertices = MAX_VERTICES) out;

uniform mat4 camera;
uniform float length;
uniform float angle;

void build_cone(vec4 position) {
    int resolution = (MAX_VERTICES - 2) / 2;
    float r0 = 0.0;
    float r1 = length * tan(angle);

    for (int i = 0; i <= resolution; i++) {

        int ind = i % resolution;
        float angle = 2 * PI * ind / resolution;
        float s = sin(angle);
        float c = cos(angle);
        vec4 v1 = position + vec4(r0 * c, r0 * s, 0.0, 1.0);
        vec4 v2 = position + vec4(r1 * c, r1 * s, length, 1.0);

        gl_Position = camera * v1;
        EmitVertex();

        gl_Position = camera * v2;
        EmitVertex();
    }
}


void main() {
    build_cone(gl_in[0].gl_Position);
}
